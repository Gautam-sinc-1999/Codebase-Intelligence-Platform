"""
Source files that are not UTF-8 (G-07).

Everything was read as UTF-8 with `errors="replace"`, turning each non-ASCII byte into U+FFFD.
`ast.parse` then failed, and the regex fallback's `[A-Za-z0-9_]+` could not match a mangled
identifier either — so the file counted toward `file_count` and `total_lines` while being
invisible to every query. Silent, and entirely ordinary in older European and Asian codebases.
"""
import os

import pytest

from app.ingestion.discovery import FileDiscovery


def _write(tmp_path, name, encoding, text):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)


# ------------------------------------------------------------------ decoding

def test_a_pep263_declaration_is_obeyed(tmp_path):
    """Python itself honours `# -*- coding: … -*-`; so must anything claiming to parse Python."""
    path = _write(tmp_path, "a.py", "latin-1",
                  "# -*- coding: latin-1 -*-\ndef a\xf1adir_caf\xe9(x):\n    return x\n")
    content, encoding, confident = FileDiscovery.read_source(path)

    assert "añadir_café" in content
    assert encoding == "latin-1" and confident is True
    assert "�" not in content, "replacement characters reached the parser"


def test_a_utf8_bom_is_consumed_not_left_in_the_source(tmp_path):
    """A leading U+FEFF makes `ast.parse` reject an otherwise perfectly ordinary file."""
    path = _write(tmp_path, "b.py", "utf-8-sig", "def with_bom():\n    return 1\n")
    content, encoding, confident = FileDiscovery.read_source(path)

    assert content.startswith("def with_bom")
    assert "﻿" not in content
    assert encoding == "utf-8-sig" and confident is True


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_wide_encodings_are_decoded(tmp_path, encoding):
    path = _write(tmp_path, f"c_{encoding}.py", encoding, "def wide_one():\n    return 1\n")
    content, detected, confident = FileDiscovery.read_source(path)

    assert "wide_one" in content
    assert confident is True and detected.startswith(encoding[:6])


def test_plain_utf8_is_read_strictly_and_confidently(tmp_path):
    path = _write(tmp_path, "d.py", "utf-8", "def café_price():\n    return 1\n")
    content, encoding, confident = FileDiscovery.read_source(path)

    assert "café_price" in content
    assert encoding == "utf-8" and confident is True


def test_an_undeclared_legacy_encoding_is_decoded_but_marked_a_guess(tmp_path):
    """
    latin-1 maps every possible byte to some character, so it never fails — and therefore never
    proves anything. Saying so is the difference between decoding and interpreting.
    """
    path = _write(tmp_path, "e.py", "cp1252",
                  "def sin_declaracion(x):\n    # coste en “euros”\n    return x\n")
    content, encoding, confident = FileDiscovery.read_source(path)

    assert "sin_declaracion" in content
    assert confident is False, "a guess was reported as a certainty"
    assert encoding in ("cp1252", "latin-1")


def test_an_empty_file_is_not_an_error(tmp_path):
    path = _write(tmp_path, "f.py", "utf-8", "")
    content, encoding, confident = FileDiscovery.read_source(path)
    assert content == "" and confident is True


# ------------------------------------------------------------------ binary classification

def test_utf16_is_not_mistaken_for_binary(tmp_path):
    """
    UTF-16 text is full of null bytes, so a naive check called it binary and dropped the file
    before anything else saw it — not merely unindexed, but absent from the file count.
    """
    path = _write(tmp_path, "wide.py", "utf-16", "def wide_one():\n    return 1\n")
    assert FileDiscovery.is_binary(path) is False


def test_genuinely_binary_content_is_still_rejected(tmp_path):
    path = tmp_path / "blob.py"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00")
    assert FileDiscovery.is_binary(str(path)) is True


def test_ordinary_text_is_not_binary(tmp_path):
    path = _write(tmp_path, "t.py", "utf-8", "def x():\n    return 1\n")
    assert FileDiscovery.is_binary(path) is False


# ------------------------------------------------------------------ end to end

def test_every_encoding_produces_symbols_when_indexed(temp_repo, tmp_path):
    """
    The property that was missing: a file that counts toward the totals must also be answerable.
    """
    import shutil
    from app.api.repositories import index_repository_folder

    root = tmp_path / "repo"
    root.mkdir()
    files = {
        "identificar.py": ("latin-1",
                           "# -*- coding: latin-1 -*-\ndef a\xf1adir_caf\xe9(x):\n    return x\n"),
        "bom.py": ("utf-8-sig", "def with_bom():\n    return 1\n"),
        "wide.py": ("utf-16", "def wide_one():\n    return 1\n"),
        "nodecl.py": ("cp1252",
                      "def sin_declaracion(x):\n    # “euros”\n    return x\n"),
        "plain.py": ("utf-8", "def normal():\n    return 1\n"),
    }
    for name, (encoding, text) in files.items():
        (root / name).write_bytes(text.encode(encoding))

    result = index_repository_folder(str(root), "enc_repo", "enc")

    assert result["file_count"] == len(files), "a file was dropped before being counted"

    by_file = {}
    for chunk in result["all_chunks"]:
        by_file.setdefault(chunk["file_path"], []).append(chunk["symbol"])

    silent = [name for name in files if not by_file.get(name)]
    assert silent == [], f"counted toward the totals but invisible to every query: {silent}"
    assert "añadir_café" in by_file["identificar.py"], "the non-ASCII identifier was mangled"


def test_an_undecodable_file_is_skipped_rather_than_counted(tmp_path, monkeypatch):
    """
    If the text genuinely cannot be recovered, omitting it is honest — counting it makes the
    repository look indexed while part of it answers nothing.
    """
    from app.api.repositories import index_repository_folder

    root = tmp_path / "repo2"
    root.mkdir()
    (root / "good.py").write_text("def good():\n    return 1\n")
    (root / "bad.py").write_text("def bad():\n    return 2\n")

    real_read = FileDiscovery.read_source

    def selective(path):
        if path.endswith("bad.py"):
            return "", "", False
        return real_read(path)

    monkeypatch.setattr(FileDiscovery, "read_source", staticmethod(selective))

    result = index_repository_folder(str(root), "skip_repo", "skip")
    assert result["file_count"] == 1, "an undecodable file was counted as indexed"
    assert {c["file_path"] for c in result["all_chunks"]} == {"good.py"}
