"""
GitHub acquisition.

**No test here touches the network.** The clone machinery is exercised against a real git
repository created in `tmp_path` and cloned from the filesystem — real git, so the argument
handling and SHA reading are genuinely tested, but with no rate limits, no outages and no
flakiness. `allow_local=True` is what opens that door, and production callers never set it.

URL validation is pure, so it is tested directly against the inputs it exists to refuse.
"""
import os
import subprocess

import pytest

from app.ingestion.git_source import (
    GitHubRepo,
    InvalidRepositoryURL,
    CloneFailed,
    RepositoryTooLarge,
    validate_github_url,
    validate_branch_name,
    derive_repository_id,
    list_remote_branches,
    remote_head_sha,
    clone_repository,
    enforce_clone_limits,
    _branch_cache,
)


# ---------------------------------------------------------------- URL validation

@pytest.mark.parametrize("url", [
    "https://github.com/psf/requests",
    "https://github.com/psf/requests.git",
    "https://github.com/psf/requests/",
    "github.com/psf/requests",                      # pasted from the address bar
    "https://github.com/psf/requests/tree/main",    # pasted from a page
    "https://GitHub.com/psf/requests",              # host case is irrelevant
])
def test_accepts_real_github_urls(url):
    repo = validate_github_url(url)
    assert repo.owner == "psf" and repo.repo == "requests"
    assert repo.clone_url == "https://github.com/psf/requests.git"


@pytest.mark.parametrize("url,because", [
    ("file:///etc/passwd", "file:// is a valid clone URL and reads the local disk"),
    ("git://github.com/a/b", "git:// is unauthenticated and unencrypted"),
    ("ssh://git@github.com/a/b", "ssh:// reaches the host's keys"),
    ("http://github.com/a/b", "plaintext"),
    ("https://github.com@evil.com/a/b", "userinfo: the real host is evil.com"),
    ("https://user:token@github.com/a/b", "credentials would leak into ps and logs"),
    ("https://github.com.evil.com/a/b", "suffix impersonation"),
    ("https://evil.com/github.com/a/b", "path impersonation"),
    ("https://github.com:8080/a/b", "explicit port is a redirection lever"),
    ("https://github.com/../../etc/a", "relative segments"),
    ("https://github.com/a/../../b", "relative segments"),
    ("--upload-pack=touch /tmp/pwned", "reads as a git option, not a URL"),
    ("--config=core.pager=sh", "reads as a git option"),
    ("https://github.com/onlyowner", "no repository name"),
    ("https://github.com/", "nothing at all"),
    ("", "empty"),
    (None, "missing"),
])
def test_rejects_hostile_urls(url, because):
    with pytest.raises(InvalidRepositoryURL):
        validate_github_url(url)


def test_userinfo_rejection_is_not_fooled_by_the_hostname_parser():
    """
    `urlsplit('https://github.com@evil.com/a/b').hostname` is 'evil.com', but this is the form
    most likely to pass a careless eye, so it gets its own assertion.
    """
    with pytest.raises(InvalidRepositoryURL) as excinfo:
        validate_github_url("https://github.com@evil.com/a/b")
    assert "credentials" in str(excinfo.value) or "userinfo" in str(excinfo.value)


@pytest.mark.parametrize("branch", ["main", "develop", "feat/checkout", "release-1.2.3", "v2"])
def test_accepts_ordinary_branch_names(branch):
    assert validate_branch_name(branch) == branch


@pytest.mark.parametrize("branch", [
    "--upload-pack=sh",   # option injection, the same trap as the URL
    "-x",
    "feat/../../etc",     # traversal
    "has space",
    "tilde~1",
    "caret^",
    "colon:",
    "question?",
    "star*",
    "bracket[",
    "back\\slash",
    "at@{brace",
    "/leading",
    "trailing/",
    "double//slash",
    "trailing.",
    "something.lock",
    "control\x01char",
    "",
])
def test_rejects_malformed_branch_names(branch):
    with pytest.raises(InvalidRepositoryURL):
        validate_branch_name(branch)


def test_branch_is_validated_when_supplied_with_the_url():
    with pytest.raises(InvalidRepositoryURL):
        validate_github_url("https://github.com/a/b", branch="--upload-pack=sh")


# ---------------------------------------------------------------- derived identity

def test_repository_id_is_stable_across_calls():
    first = derive_repository_id("psf", "requests", "main")
    assert first == derive_repository_id("psf", "requests", "main")
    assert first.startswith("gh_") and len(first) == 15


def test_repository_id_is_case_insensitive_like_github():
    """`Owner/Repo` and `owner/repo` are one repository; they must not become two indexes."""
    assert derive_repository_id("PSF", "Requests", "main") == \
           derive_repository_id("psf", "requests", "main")


def test_repository_id_distinguishes_branches_and_repositories():
    ids = {
        derive_repository_id("psf", "requests", "main"),
        derive_repository_id("psf", "requests", "develop"),
        derive_repository_id("psf", "other", "main"),
        derive_repository_id("other", "requests", "main"),
    }
    assert len(ids) == 4


# ---------------------------------------------------------------- local git fixture

def _git(*args, cwd):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    return subprocess.run(["git", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)


@pytest.fixture
def origin_repo(tmp_path):
    """A real git repository on disk, with three branches, to clone from."""
    root = tmp_path / "origin"
    root.mkdir()
    _git("init", "-b", "main", cwd=root)

    (root / "app.py").write_text("def handler(request):\n    return validate(request)\n")
    (root / "README.md").write_text("# origin\n")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "initial", cwd=root)

    _git("checkout", "-b", "develop", cwd=root)
    (root / "extra.py").write_text("def extra():\n    return 2\n")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "develop work", cwd=root)

    _git("checkout", "-b", "feat/checkout", cwd=root)
    (root / "checkout.py").write_text("def checkout():\n    return 3\n")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "checkout work", cwd=root)

    _git("checkout", "main", cwd=root)
    return root


@pytest.fixture(autouse=True)
def _clear_branch_cache():
    _branch_cache.clear()
    yield
    _branch_cache.clear()


# ---------------------------------------------------------------- remote inspection

def test_lists_every_branch_with_its_sha(origin_repo):
    result = list_remote_branches(str(origin_repo), allow_local=True)
    names = {b["name"] for b in result["branches"]}
    assert names == {"main", "develop", "feat/checkout"}
    assert all(len(b["commit_sha"]) == 40 for b in result["branches"])


def test_default_branch_is_identified_and_sorted_first(origin_repo):
    result = list_remote_branches(str(origin_repo), allow_local=True)
    assert result["default_branch"] == "main"
    assert result["branches"][0]["name"] == "main"
    assert result["branches"][0]["is_default"] is True


def test_branch_listing_is_cached(origin_repo):
    """The UI calls this on every paste; branch lists do not change by the second."""
    first = list_remote_branches(str(origin_repo), allow_local=True)

    _git("checkout", "-b", "brand-new", cwd=origin_repo)
    _git("commit", "--allow-empty", "-m", "new branch", cwd=origin_repo)

    cached = list_remote_branches(str(origin_repo), allow_local=True)
    assert cached == first, "cache was bypassed"

    fresh = list_remote_branches(str(origin_repo), allow_local=True, use_cache=False)
    assert "brand-new" in {b["name"] for b in fresh["branches"]}


def test_remote_head_sha_tracks_new_commits(origin_repo):
    """The primitive behind drift detection: it must never report a stale SHA."""
    before = remote_head_sha(str(origin_repo), "main", allow_local=True)

    (origin_repo / "app.py").write_text("def handler(request):\n    return 'changed'\n")
    _git("add", ".", cwd=origin_repo)
    _git("commit", "-m", "second", cwd=origin_repo)

    after = remote_head_sha(str(origin_repo), "main", allow_local=True)
    assert before != after and len(after) == 40


def test_remote_head_sha_is_none_for_an_unknown_branch(origin_repo):
    assert remote_head_sha(str(origin_repo), "no-such-branch", allow_local=True) is None


def test_listing_a_nonexistent_remote_fails_cleanly(tmp_path):
    with pytest.raises(CloneFailed):
        list_remote_branches(str(tmp_path / "nowhere"), allow_local=True)


# ---------------------------------------------------------------- cloning

def test_clone_produces_a_working_tree_and_a_sha(origin_repo, tmp_path):
    dest = tmp_path / "work"
    sha = clone_repository(str(origin_repo), "main", str(dest), allow_local=True)

    assert len(sha) == 40
    assert (dest / "app.py").exists()
    assert "def handler" in (dest / "app.py").read_text()


def test_clone_checks_out_the_requested_branch(origin_repo, tmp_path):
    dest = tmp_path / "work"
    clone_repository(str(origin_repo), "feat/checkout", str(dest), allow_local=True)

    assert (dest / "checkout.py").exists(), "cloned the wrong branch"
    assert (dest / "extra.py").exists()


def test_clone_is_shallow(origin_repo, tmp_path):
    """
    History is never read downstream, and for a large repository it is most of the download.

    Cloned through a `file://` URL rather than a bare path: git silently ignores `--depth` for
    local-path clones (it hardlinks the object store instead) and says so —
    "--depth is ignored in local clones; use file:// instead". A bare path would therefore
    report two commits and prove nothing about the flag the production path relies on.
    """
    dest = tmp_path / "work"
    clone_repository(f"file://{origin_repo}", "develop", str(dest), allow_local=True)

    log = subprocess.run(["git", "-C", str(dest), "rev-list", "--count", "HEAD"],
                         capture_output=True, text=True, check=True)
    assert log.stdout.strip() == "1", "clone was not shallow"
    assert (dest / "extra.py").exists(), "shallow clone lost the branch's content"


def test_clone_of_a_missing_branch_fails_and_leaves_no_directory(origin_repo, tmp_path):
    dest = tmp_path / "work"
    with pytest.raises(CloneFailed):
        clone_repository(str(origin_repo), "no-such-branch", str(dest), allow_local=True)
    assert not dest.exists(), "a failed clone left a partial directory behind"


def test_clone_replaces_an_existing_destination(origin_repo, tmp_path):
    dest = tmp_path / "work"
    dest.mkdir()
    (dest / "stale.py").write_text("# from a previous clone\n")

    clone_repository(str(origin_repo), "main", str(dest), allow_local=True)
    assert not (dest / "stale.py").exists()


def test_production_clone_path_refuses_a_local_url(origin_repo, tmp_path):
    """
    The validator is not the only guard.

    With `allow_local=False` the URL is re-validated inside the clone function, so a caller that
    forgets to validate cannot turn this into an arbitrary-path clone.
    """
    with pytest.raises(InvalidRepositoryURL):
        clone_repository(str(origin_repo), "main", str(tmp_path / "w"), allow_local=False)


def test_git_never_prompts_for_credentials(tmp_path):
    """
    Pointed at something needing auth, git must fail fast rather than block on a prompt.

    Without GIT_TERMINAL_PROMPT=0 this hangs until the timeout, turning a clear error into a
    five-minute stall.
    """
    missing = tmp_path / "definitely-not-a-repo"
    missing.mkdir()
    with pytest.raises(CloneFailed):
        clone_repository(str(missing), "main", str(tmp_path / "w"), allow_local=True)


# ---------------------------------------------------------------- limits

def test_limits_accept_an_ordinary_repository(origin_repo, tmp_path):
    dest = tmp_path / "work"
    clone_repository(str(origin_repo), "main", str(dest), allow_local=True)
    stats = enforce_clone_limits(str(dest))
    assert stats["file_count"] >= 2 and stats["total_bytes"] > 0


def test_limits_reject_too_many_files(origin_repo, tmp_path, monkeypatch):
    from app.core.config import settings

    dest = tmp_path / "work"
    clone_repository(str(origin_repo), "main", str(dest), allow_local=True)
    monkeypatch.setattr(settings, "MAX_ARCHIVE_ENTRIES", 1)

    with pytest.raises(RepositoryTooLarge):
        enforce_clone_limits(str(dest))


def test_limits_reject_an_oversized_tree(origin_repo, tmp_path, monkeypatch):
    from app.core.config import settings

    dest = tmp_path / "work"
    clone_repository(str(origin_repo), "main", str(dest), allow_local=True)
    monkeypatch.setattr(settings, "MAX_TOTAL_UNCOMPRESSED_BYTES", 1)

    with pytest.raises(RepositoryTooLarge):
        enforce_clone_limits(str(dest))


def test_git_metadata_does_not_count_toward_the_limits(origin_repo, tmp_path):
    """
    A clone carries `.git`, which discovery ignores.

    Counting it would reject ordinary repositories — the entry-count cap already had to be
    narrowed once for exactly this reason on the zip path (F-40).
    """
    dest = tmp_path / "work"
    clone_repository(str(origin_repo), "main", str(dest), allow_local=True)

    assert (dest / ".git").is_dir(), "precondition: the clone has git metadata"

    counted = enforce_clone_limits(str(dest))["file_count"]
    on_disk = sum(len(files) for _, _, files in os.walk(dest))
    assert counted < on_disk, "git metadata was counted toward the limits"
    assert counted == 2, "expected exactly the two tracked files"


def test_option_injection_cannot_execute_even_if_validation_is_bypassed(tmp_path):
    """
    Defence in depth: the '--' separator, not just the validator.

    `git clone --upload-pack=<cmd>` runs <cmd>. Validation rejects such a string long before git
    sees it, but a guard that only works when an earlier guard worked is one refactor away from
    being no guard at all. Here validation is deliberately skipped (`allow_local=True`) to prove
    the separator alone prevents execution.
    """
    marker = tmp_path / "pwned"
    hostile = f"--upload-pack=touch {marker}"

    with pytest.raises(CloneFailed):
        clone_repository(hostile, "main", str(tmp_path / "w"), allow_local=True)

    assert not marker.exists(), "option injection executed — the '--' separator did not hold"


def test_the_url_dash_guard_is_redundant_but_deliberate():
    """
    A leading '-' is already caught by the https check, since it parses to no scheme.

    Asserted so the redundancy is recorded as intentional: it is the guard that keeps working if
    the parsing above is ever relaxed to accept more URL shapes.
    """
    from urllib.parse import urlsplit
    assert urlsplit("--upload-pack=sh").scheme == ""

    with pytest.raises(InvalidRepositoryURL) as excinfo:
        validate_github_url("--upload-pack=sh")
    assert "'-'" in str(excinfo.value), "expected the dash guard, not the scheme guard, to fire first"
