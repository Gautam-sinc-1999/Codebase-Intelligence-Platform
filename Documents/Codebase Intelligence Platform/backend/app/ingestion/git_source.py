"""
Acquiring a repository from GitHub.

This module's only job is to turn a pasted URL into a directory on disk. Everything downstream —
discovery, parsing, embedding, graphing — already takes a plain directory and neither knows nor
cares how it got there, which is what makes git an alternative *acquisition* step rather than a
second pipeline.

**The split matters.** `validate_github_url` is pure and does no I/O; `clone_repository` does I/O
and validates again. That separation is what lets the clone machinery be tested offline against a
local repository (`allow_local=True`) while the production path still refuses anything that is not
an `https://github.com/owner/repo` URL — the validator is not the only guard, so a caller that
forgets it does not open a hole.

A clone is remote-controlled disk writing, exactly like extracting an uploaded zip, and is treated
with the same suspicion (see `_safe_extract_zip` in app/api/repositories.py).
"""
import os
import re
import time
import shutil
import hashlib
import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from app.core.config import settings
from app.ingestion.discovery import IGNORE_DIRS

logger = logging.getLogger("ingestion.git_source")

GITHUB_HOST = "github.com"

# How long a git subprocess may run before it is killed. A slow or hostile remote must not be able
# to pin a worker indefinitely.
LS_REMOTE_TIMEOUT = 30
CLONE_TIMEOUT = 300

# Branch listings are requested on every paste and do not change by the second.
BRANCH_CACHE_TTL = 300


class GitSourceError(Exception):
    """Base for every failure in this module."""


class InvalidRepositoryURL(GitSourceError):
    """The URL is not a GitHub repository URL we are willing to clone."""


class CloneFailed(GitSourceError):
    """git exited non-zero, timed out, or produced nothing usable."""


class RepositoryTooLarge(GitSourceError):
    """The cloned tree exceeds the limits that apply to uploads."""


# GitHub owner names: alphanumeric and hyphens, no leading or trailing hyphen, <= 39 characters.
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
# Repository names additionally allow dots and underscores. '..' is excluded separately, since a
# dot is legitimate in a repository name (`foo.js`) but a doubled one is a traversal attempt.
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

# Characters git itself forbids in a ref name, plus whitespace.
_BRANCH_FORBIDDEN = set(" \t\n\r~^:?*[\\\x7f")


@dataclass(frozen=True)
class GitHubRepo:
    """A validated GitHub repository reference."""
    owner: str
    repo: str
    clone_url: str
    branch: Optional[str] = None

    @property
    def name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def display_name(self) -> str:
        return f"{self.name}@{self.branch}" if self.branch else self.name


def validate_github_url(raw: str, branch: Optional[str] = None) -> GitHubRepo:
    """
    Parses and validates a pasted GitHub URL. Pure — no network, no disk.

    Rejects, with the reason each one matters:

    - non-`https` schemes — `file:///etc` is a perfectly valid clone URL, and `ssh://` and
      `git://` reach hosts and credentials that a pasted string should not command
    - **userinfo** — `https://github.com@evil.com/a/b` has host `evil.com`, not github.com. This
      is the trap that looks most like a legitimate URL
    - credentials — `https://user:token@github.com/...` would leak a token into `ps` and logs
    - any host that is not exactly `github.com` — including `github.com.evil.com` and
      `evil.com/github.com/...`
    - explicit ports — the real host does not need one, and it is a redirection lever
    - `..` in any segment, and names outside GitHub's own character set
    - a leading `-` anywhere, which git would read as a command-line option
    """
    if not raw or not isinstance(raw, str):
        raise InvalidRepositoryURL("No repository URL was provided.")

    candidate = raw.strip()
    if not candidate:
        raise InvalidRepositoryURL("No repository URL was provided.")

    # A value starting with '-' is an option to every command-line tool, not a URL. Rejected
    # before parsing so it can never reach argv, belt-and-braces with the '--' separator below.
    if candidate.startswith("-"):
        raise InvalidRepositoryURL("A repository URL cannot begin with '-'.")

    # Accept the bare `github.com/owner/repo` form users paste from the address bar.
    if candidate.startswith(f"{GITHUB_HOST}/"):
        candidate = f"https://{candidate}"

    parts = urlsplit(candidate)

    if parts.scheme != "https":
        raise InvalidRepositoryURL(
            f"Only https:// GitHub URLs are supported (got '{parts.scheme or 'no scheme'}')."
        )

    # `parts.hostname` lowercases and strips userinfo, so compare the raw netloc too — the
    # difference between them is exactly where the impersonation lives.
    if "@" in parts.netloc:
        raise InvalidRepositoryURL(
            "A repository URL must not contain credentials or a userinfo section."
        )
    if parts.port is not None:
        raise InvalidRepositoryURL("A repository URL must not specify a port.")
    if (parts.hostname or "").lower() != GITHUB_HOST:
        raise InvalidRepositoryURL(
            f"Only github.com repositories are supported (got '{parts.hostname or 'no host'}')."
        )

    segments = [s for s in parts.path.split("/") if s]
    if any(s == ".." or s == "." for s in segments):
        raise InvalidRepositoryURL("A repository URL must not contain relative path segments.")
    if len(segments) < 2:
        raise InvalidRepositoryURL(
            "Expected a URL of the form https://github.com/owner/repository."
        )

    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]

    if not _OWNER_RE.match(owner):
        raise InvalidRepositoryURL(f"'{owner}' is not a valid GitHub owner name.")
    if not _REPO_RE.match(repo) or ".." in repo:
        raise InvalidRepositoryURL(f"'{repo}' is not a valid GitHub repository name.")

    # Extra path beyond owner/repo (/tree/main, /blob/...) is tolerated and dropped: people paste
    # the page they were looking at. The branch is taken from the `branch` argument, never
    # inferred from the URL, so that what is cloned is always what the caller asked for.
    validated_branch = validate_branch_name(branch) if branch else None

    return GitHubRepo(
        owner=owner,
        repo=repo,
        clone_url=f"https://{GITHUB_HOST}/{owner}/{repo}.git",
        branch=validated_branch,
    )


def validate_branch_name(name: str) -> str:
    """
    Validates a branch name against git's own ref rules.

    The branch reaches `git clone --branch <name>`, so a name beginning with '-' is option
    injection in the same way a URL is. The remaining rules are `git check-ref-format`'s: they
    prevent a name that git would reject or, worse, interpret.
    """
    if not name or not isinstance(name, str):
        raise InvalidRepositoryURL("No branch name was provided.")

    branch = name.strip()
    if not branch:
        raise InvalidRepositoryURL("No branch name was provided.")
    if branch.startswith("-"):
        raise InvalidRepositoryURL("A branch name cannot begin with '-'.")
    if len(branch) > 255:
        raise InvalidRepositoryURL("Branch name is too long.")
    if any(c in _BRANCH_FORBIDDEN or ord(c) < 0x20 for c in branch):
        raise InvalidRepositoryURL(f"Branch name '{name}' contains characters git forbids.")
    if ".." in branch or "@{" in branch:
        raise InvalidRepositoryURL(f"Branch name '{name}' contains a forbidden sequence.")
    if branch.startswith("/") or branch.endswith("/") or "//" in branch:
        raise InvalidRepositoryURL(f"Branch name '{name}' is not a well-formed ref.")
    if branch.endswith(".") or branch.endswith(".lock"):
        raise InvalidRepositoryURL(f"Branch name '{name}' is not a well-formed ref.")

    return branch


def derive_repository_id(owner: str, repo: str, branch: str) -> str:
    """
    A stable id derived from the repository identity, rather than a fresh uuid per import.

    Zip uploads mint `repo_<uuid4>`, so uploading the same project twice creates two unrelated
    repositories. For git that would be wrong: re-syncing must *update* what is already indexed.
    Branch is part of the identity because each branch is independently indexed.

    Owner and repo are lower-cased because GitHub treats them case-insensitively, so
    `Owner/Repo` and `owner/repo` must not become two indexes of one thing.
    """
    identity = f"{GITHUB_HOST}/{owner.lower()}/{repo.lower()}@{branch}"
    return "gh_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- subprocess plumbing


def _git_env(allow_local: bool = False) -> Dict[str, str]:
    """
    A git environment that cannot block, prompt, or be steered by ambient configuration.

    `GIT_TERMINAL_PROMPT=0` is the important one: pointed at a private repository, git otherwise
    blocks on a username prompt and the request hangs until the timeout rather than failing
    immediately with a usable message.

    `GIT_ALLOW_PROTOCOL` is defence in depth — even if a non-https URL reached this function
    despite validation, git itself would refuse to speak the protocol.
    """
    env = dict(os.environ)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
        "SSH_ASKPASS": "",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ALLOW_PROTOCOL": "https:file" if allow_local else "https",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    for leaked in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(leaked, None)
    return env


def _run_git(args: List[str], timeout: int, allow_local: bool = False) -> subprocess.CompletedProcess:
    """Runs git with an argument list — never a shell string, so nothing can be word-split."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_git_env(allow_local),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise CloneFailed(f"git timed out after {timeout}s.") from e
    except FileNotFoundError as e:
        raise CloneFailed("git is not installed or not on PATH.") from e


def _sanitize_git_error(stderr: str, url: str) -> str:
    """Trims git's output to one useful line and keeps the URL out of it."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    message = lines[-1] if lines else "git failed with no output."
    return message.replace(url, "<url>")[:300]


# ---------------------------------------------------------------- remote inspection

# (url -> (expires_at, payload)). Small enough that eviction is not worth the code.
_branch_cache: Dict[str, tuple] = {}


def list_remote_branches(url: str, *, use_cache: bool = True,
                         allow_local: bool = False) -> Dict[str, object]:
    """
    Lists a repository's branches without cloning it.

    `git ls-remote` costs about a second and needs no token for public repositories, where the
    GitHub REST API would need one to escape a 60/hour anonymous rate limit.
    """
    if allow_local:
        target = url
    else:
        target = validate_github_url(url).clone_url

    if use_cache:
        cached = _branch_cache.get(target)
        if cached and cached[0] > time.monotonic():
            return cached[1]

    heads = _run_git(["ls-remote", "--heads", "--", target], LS_REMOTE_TIMEOUT, allow_local)
    if heads.returncode != 0:
        raise CloneFailed(_sanitize_git_error(heads.stderr, target))

    branches = []
    for line in heads.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if not ref.startswith("refs/heads/"):
            continue
        branches.append({"name": ref[len("refs/heads/"):], "commit_sha": sha.strip()})

    default_branch = _default_branch(target, allow_local) or ""
    # Fall back to whatever exists when the remote reports no symref (bare local repos often
    # do not), so the caller always has something selectable.
    if not default_branch and branches:
        default_branch = next(
            (b["name"] for b in branches if b["name"] in ("main", "master")),
            branches[0]["name"],
        )

    for branch in branches:
        branch["is_default"] = branch["name"] == default_branch

    branches.sort(key=lambda b: (not b["is_default"], b["name"]))
    payload = {"default_branch": default_branch, "branches": branches}

    if use_cache:
        _branch_cache[target] = (time.monotonic() + BRANCH_CACHE_TTL, payload)
    return payload


def _default_branch(target: str, allow_local: bool = False) -> Optional[str]:
    """Reads the remote's HEAD symref, which is what 'default branch' means to git."""
    result = _run_git(["ls-remote", "--symref", "--", target, "HEAD"],
                      LS_REMOTE_TIMEOUT, allow_local)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("ref:"):
            ref = line.split()[1]
            if ref.startswith("refs/heads/"):
                return ref[len("refs/heads/"):]
    return None


def remote_head_sha(url: str, branch: str, *, allow_local: bool = False,
                    use_cache: bool = False) -> Optional[str]:
    """
    The current SHA of one branch — the primitive behind both sync and drift detection.

    **Uncached by default, deliberately.** `/sync` uses this to decide whether any work is needed
    at all, and a stale answer there would skip a sync the user actually asked for.

    `use_cache=True` is for the drift check, which is called every time a thread is opened. There
    the cost of staleness is only that a new commit is noticed up to `BRANCH_CACHE_TTL` late —
    a freshness question, not a correctness one — and the saving is a network round trip per
    thread click.
    """
    target = url if allow_local else validate_github_url(url).clone_url
    ref = validate_branch_name(branch)

    cache_key = f"sha::{target}::{ref}"
    if use_cache:
        cached = _branch_cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return cached[1]

    result = _run_git(["ls-remote", "--heads", "--", target, ref],
                      LS_REMOTE_TIMEOUT, allow_local)
    if result.returncode != 0:
        raise CloneFailed(_sanitize_git_error(result.stderr, target))

    found_sha = None
    for line in result.stdout.splitlines():
        sha, _, found = line.partition("\t")
        if found.strip() == f"refs/heads/{ref}":
            found_sha = sha.strip()
            break

    if use_cache:
        _branch_cache[cache_key] = (time.monotonic() + BRANCH_CACHE_TTL, found_sha)
    return found_sha


# ---------------------------------------------------------------- cloning


def clone_repository(url: str, branch: str, dest: str, *, allow_local: bool = False) -> str:
    """
    Shallow-clones one branch into `dest` and returns its commit SHA.

    `--depth 1 --single-branch --no-tags`: history is never read by anything downstream, and for
    a large repository history is most of the download.

    `allow_local` exists for tests, which clone from a repository created in a tmp directory —
    real git, no network, no rate limits, no flakiness. It is never set by production callers, and
    when it is false the URL is re-validated here rather than trusted from the caller.
    """
    if not allow_local:
        url = validate_github_url(url).clone_url
    ref = validate_branch_name(branch)

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    result = _run_git(
        [
            "-c", "credential.helper=",           # never consult a stored credential
            "clone", "--depth", "1", "--single-branch", "--no-tags",
            "--branch", ref,
            "--", url, dest,                      # '--' so neither value can be read as a flag
        ],
        CLONE_TIMEOUT,
        allow_local,
    )

    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed(_sanitize_git_error(result.stderr, url))

    head = _run_git(["-C", dest, "rev-parse", "HEAD"], LS_REMOTE_TIMEOUT, allow_local)
    if head.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed("Cloned, but could not read the commit SHA.")

    return head.stdout.strip()


def enforce_clone_limits(path: str) -> Dict[str, int]:
    """
    Applies the upload limits to a cloned tree.

    A repository can be as hostile as an archive — thousands of generated files, or one enormous
    blob — and the caps that protect the zip path (`MAX_ARCHIVE_ENTRIES`,
    `MAX_TOTAL_UNCOMPRESSED_BYTES`) have no reason to stop applying because the bytes arrived over
    git instead.

    Checked after the clone rather than during it, so a repository can in principle occupy disk
    briefly before being rejected; `--depth 1` bounds that, and a true quota would need filesystem
    support rather than application code.
    """
    entries = 0
    total_bytes = 0

    for root, dirnames, filenames in os.walk(path):
        # `.git` is the clone's own metadata and node_modules is never indexed; counting either
        # toward the caps would reject ordinary repositories.
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and d != ".git"]

        for filename in filenames:
            entries += 1
            if entries > settings.MAX_ARCHIVE_ENTRIES:
                raise RepositoryTooLarge(
                    f"Repository contains more than {settings.MAX_ARCHIVE_ENTRIES} files."
                )
            try:
                total_bytes += os.path.getsize(os.path.join(root, filename))
            except OSError:
                continue
            if total_bytes > settings.MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise RepositoryTooLarge(
                    f"Repository exceeds "
                    f"{settings.MAX_TOTAL_UNCOMPRESSED_BYTES // (1024 * 1024)} MB."
                )

    return {"file_count": entries, "total_bytes": total_bytes}
