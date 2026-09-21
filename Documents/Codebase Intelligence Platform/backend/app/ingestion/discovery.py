import os
import codecs
import re
import hashlib
import logging
from typing import List, Dict, Any, Iterator

from app.core.config import settings

logger = logging.getLogger("ingestion.discovery")

# Directories never worth indexing. Listed explicitly rather than excluding every dot-directory,
# which also swallowed .github/ and other configuration developers ask about.
IGNORE_DIRS = {
    "node_modules", "target", "dist", "build", "coverage", "vendor", "bower_components",
    "venv", "env", "__pycache__", ".git", ".hg", ".svn", ".idea", ".vscode", ".venv",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".cache", ".next",
    ".nuxt", ".svelte-kit", ".parcel-cache", ".gradle", ".terraform", ".serverless",
    ".egg-info", ".ipynb_checkpoints", "__MACOSX", ".DS_Store",
}

IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin", ".tar", ".gz", ".bz2",
    ".7z", ".rar", ".zip", ".jar", ".war", ".class", ".o", ".a", ".lib", ".wasm",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp", ".webp", ".pdf", ".psd",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".webm",
    ".db", ".sqlite", ".sqlite3", ".lock",
}

# Files that must never be indexed regardless of extension.
#
# Uploaded archives routinely contain real credentials — this repository's own data directory
# holds third-party .env files and private keys. Indexing them would copy secrets into the chunk
# store, the vector embeddings and, via retrieval, into LLM prompts sent to an external provider.
SECRET_FILE_PATTERNS = [
    # Matches .env, .env.local, .env.production — but NOT the committed templates, which carry
    # key *names* without values and are a legitimate thing to ask about.
    re.compile(r"^\.env(?!\.(example|sample|template|dist)$)($|\.)", re.IGNORECASE),
    re.compile(r"^.*\.(pem|key|p12|pfx|keystore|jks)$", re.IGNORECASE),
    re.compile(r"^(id_rsa|id_dsa|id_ecdsa|id_ed25519)(\.pub)?$", re.IGNORECASE),
    re.compile(r"^\.(npmrc|pypirc|netrc|htpasswd)$", re.IGNORECASE),
    re.compile(r"^(credentials|secrets?)\.(json|ya?ml|ini|cfg)$", re.IGNORECASE),
    re.compile(r"^service-account.*\.json$", re.IGNORECASE),
]

# Dotfiles worth indexing: developers genuinely ask how CI, linting and containers are set up.
ALLOWED_DOTFILES = {
    ".gitignore", ".dockerignore", ".gitattributes", ".editorconfig", ".env.example",
    ".env.sample", ".env.template", ".env.dist", ".eslintrc", ".eslintrc.js", ".eslintrc.json",
    ".prettierrc", ".babelrc", ".nvmrc", ".python-version", ".ruff.toml", ".flake8",
}

LANGUAGE_MAP = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "react_jsx",
    ".ts": "typescript",
    ".tsx": "react_tsx",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sh": "shell",
    ".toml": "toml",
}


class FileDiscovery:
    """
    Discovers files in a code repository, detecting language, file path, size, and content hashes.
    """

    # Byte-order marks, longest first so UTF-32's is not mistaken for UTF-16's prefix.
    _BOMS = (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    )

    # PEP 263: `# -*- coding: latin-1 -*-`, honoured by Python itself in the first two lines.
    _CODING_RE = re.compile(rb"coding[:=]\s*([-\w.]+)")

    @staticmethod
    def is_binary(file_path: str) -> bool:
        """
        Whether a file is binary, by looking for null bytes.

        UTF-16 and UTF-32 text is full of null bytes, so a naive check calls it binary and the
        file is dropped before anything else sees it — not merely unindexed but absent from the
        file count as well. A byte-order mark says the opposite, and is checked first.
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
        except Exception:
            return True

        for bom, _ in FileDiscovery._BOMS:
            if chunk.startswith(bom):
                return False

        return b"\x00" in chunk

    @classmethod
    def read_source(cls, file_path: str) -> tuple:
        """
        Reads a source file as text, returning `(content, encoding, confident)`.

        Reading everything as UTF-8 with `errors="replace"` turned every non-ASCII byte into
        U+FFFD. `ast.parse` then failed, and the regex fallback's `[A-Za-z0-9_]+` could not match
        a mangled identifier either — so a latin-1 file counted toward `file_count` and
        `total_lines` while being invisible to every query. Silent, and entirely ordinary in older
        European and Asian codebases.

        Resolution order, most authoritative first:

        1. a byte-order mark, which states the encoding outright
        2. a PEP 263 `# -*- coding: … -*-` declaration, which Python itself obeys
        3. UTF-8, strictly — the overwhelming majority, and a strict attempt is what makes the
           difference between knowing and guessing
        4. cp1252, then latin-1

        `confident` is False only for step 4, where the bytes are being interpreted rather than
        decoded: latin-1 maps every possible byte to some character, so it never fails and
        therefore never proves anything.
        """
        try:
            with open(file_path, "rb") as handle:
                raw = handle.read()
        except Exception:
            return "", "", False

        if not raw:
            return "", "utf-8", True

        for bom, encoding in cls._BOMS:
            if raw.startswith(bom):
                try:
                    # utf-8-sig, utf-16 and utf-32 each consume their own BOM. Left in place it
                    # becomes a U+FEFF at position 0 and `ast.parse` rejects the file.
                    return raw.decode(encoding), encoding, True
                except UnicodeDecodeError:
                    break

        declared = None
        for line in raw.split(b"\n", 2)[:2]:
            match = cls._CODING_RE.search(line)
            if match:
                declared = match.group(1).decode("ascii", "ignore")
                break
        if declared:
            try:
                return raw.decode(declared), declared, True
            except (UnicodeDecodeError, LookupError):
                pass

        try:
            return raw.decode("utf-8"), "utf-8", True
        except UnicodeDecodeError:
            pass

        for encoding in ("cp1252", "latin-1"):
            try:
                return raw.decode(encoding), encoding, False
            except UnicodeDecodeError:
                continue

        # Unreachable in practice — latin-1 decodes any byte sequence — but a caller must never
        # be handed replacement characters believing them to be source.
        return "", "", False

    @staticmethod
    def is_secret_file(file_name: str) -> bool:
        return any(pattern.match(file_name) for pattern in SECRET_FILE_PATTERNS)

    @classmethod
    def should_index(cls, file_name: str) -> bool:
        """Decides whether a file is worth indexing, by name alone."""
        if cls.is_secret_file(file_name):
            return False

        extension = os.path.splitext(file_name)[1].lower()
        if extension in IGNORE_EXTENSIONS:
            return False

        if file_name.startswith("."):
            # Previously every dotfile was skipped, hiding CI and tooling configuration.
            return file_name.lower() in ALLOWED_DOTFILES

        return True

    @classmethod
    def iter_repository(cls, repo_path: str, repo_id: str) -> Iterator[Dict[str, Any]]:
        """
        Yields one file at a time.

        Discovery previously built a list holding the full text of every file before parsing
        began, so peak memory scaled with the whole repository. Streaming means only the file
        being parsed is resident.
        """
        repo_path = os.path.abspath(repo_path)
        max_bytes = settings.MAX_INDEXED_FILE_BYTES

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.endswith(".egg-info")]

            for file_name in files:
                if not cls.should_index(file_name):
                    continue

                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, repo_path)

                try:
                    # Checked before reading, so an oversized file is never loaded at all.
                    size = os.path.getsize(full_path)
                except OSError:
                    continue

                if size > max_bytes:
                    logger.info(
                        "Skipping %s: %d bytes exceeds the %d byte indexing limit.",
                        rel_path, size, max_bytes,
                    )
                    continue

                if cls.is_binary(full_path):
                    continue

                try:
                    content, encoding, confident = cls.read_source(full_path)
                    if not content and os.path.getsize(full_path) > 0:
                        # Counting a file that yielded nothing makes the repository look indexed
                        # while part of it is invisible. Skipping it is honest; the log says which.
                        logger.warning("Skipping %s: its text could not be decoded.", rel_path)
                        continue
                    if not confident and encoding:
                        logger.info(
                            "Read %s as %s (guessed — no BOM, no coding declaration, not UTF-8).",
                            rel_path, encoding,
                        )
                except Exception as e:
                    logger.warning("Skipping %s due to read error: %s", rel_path, e)
                    continue

                extension = os.path.splitext(file_name)[1].lower()
                yield {
                    "repository_id": repo_id,
                    "relative_path": rel_path,
                    "absolute_path": full_path,
                    "file_name": file_name,
                    "extension": extension,
                    "language": LANGUAGE_MAP.get(extension, "unknown"),
                    "size_bytes": len(content),
                    "line_count": len(content.splitlines()),
                    "file_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "content": content,
                }

    @classmethod
    def discover_repository(cls, repo_path: str, repo_id: str) -> List[Dict[str, Any]]:
        """Materialises every discovered file. Prefer iter_repository() for indexing."""
        return list(cls.iter_repository(repo_path, repo_id))
