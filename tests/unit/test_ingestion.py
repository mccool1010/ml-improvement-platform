"""Tests for dataset acquisition.

The raw file is not committed, so integrity checking is what makes a result
attributable to specific bytes. A changed upstream mirror must fail the run, not
silently alter every metric downstream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml_platform.data import ingestion


def _write(path: Path, content: bytes = b"col\n1\n2\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


class TestChecksums:
    def test_the_digest_matches_hashlib(self, tmp_path: Path) -> None:
        expected = _write(tmp_path / "f.csv")
        assert ingestion.sha256(tmp_path / "f.csv") == expected

    def test_large_files_stream_rather_than_load(self, tmp_path: Path) -> None:
        """The real file is 179 MB, so hashing must not read it into memory."""
        content = b"x" * (3 * (1 << 20))
        expected = _write(tmp_path / "big.bin", content)
        assert ingestion.sha256(tmp_path / "big.bin") == expected

    def test_a_one_byte_change_changes_the_digest(self, tmp_path: Path) -> None:
        first = _write(tmp_path / "a.csv", b"col\n1\n")
        second = _write(tmp_path / "b.csv", b"col\n2\n")
        assert first != second


class TestVerify:
    def test_a_matching_checksum_passes(self, tmp_path: Path) -> None:
        expected = _write(tmp_path / "f.csv")
        ingestion.verify(tmp_path / "f.csv", expected)

    def test_a_mismatched_checksum_raises(self, tmp_path: Path) -> None:
        """The guard against a silently changed upstream mirror."""
        _write(tmp_path / "f.csv")
        with pytest.raises(ingestion.DataIntegrityError, match="checksum mismatch"):
            ingestion.verify(tmp_path / "f.csv", "0" * 64)

    def test_the_error_names_the_file_and_both_digests(self, tmp_path: Path) -> None:
        expected = _write(tmp_path / "register.csv")
        with pytest.raises(ingestion.DataIntegrityError) as excinfo:
            ingestion.verify(tmp_path / "register.csv", "0" * 64)
        message = str(excinfo.value)
        assert "register.csv" in message
        assert expected in message


class TestDownload:
    def test_an_existing_file_is_not_refetched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "f.csv"
        _write(destination)

        def _explode(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("network access attempted for a cached file")

        monkeypatch.setattr(ingestion.urllib.request, "urlopen", _explode)
        assert ingestion.download("http://example.invalid/f.csv", destination) == destination

    def test_force_refetches_even_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "f.csv"
        _write(destination, b"old")
        monkeypatch.setattr(ingestion.urllib.request, "urlopen", _fake_urlopen(b"new"))

        ingestion.download("http://example.invalid/f.csv", destination, force=True)
        assert destination.read_bytes() == b"new"

    def test_missing_parent_directories_are_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "a" / "b" / "f.csv"
        monkeypatch.setattr(ingestion.urllib.request, "urlopen", _fake_urlopen(b"data"))
        ingestion.download("http://example.invalid/f.csv", destination)
        assert destination.read_bytes() == b"data"

    def test_no_partial_file_is_left_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Downloads stage then rename, so an interrupted run cannot be mistaken
        for a complete one."""
        destination = tmp_path / "f.csv"
        monkeypatch.setattr(ingestion.urllib.request, "urlopen", _fake_urlopen(b"data"))
        ingestion.download("http://example.invalid/f.csv", destination)
        assert not list(tmp_path.glob("*.partial"))


class TestAcquire:
    def test_a_cached_verified_file_is_returned(self, tmp_path: Path) -> None:
        destination = tmp_path / "f.csv"
        checksum = _write(destination)
        source = {"url": "http://example.invalid/f.csv", "sha256": checksum}
        assert ingestion.acquire(source, destination) == destination

    def test_a_corrupt_cached_file_fails_rather_than_being_used(self, tmp_path: Path) -> None:
        destination = tmp_path / "f.csv"
        _write(destination)
        source = {"url": "http://example.invalid/f.csv", "sha256": "0" * 64}
        with pytest.raises(ingestion.DataIntegrityError):
            ingestion.acquire(source, destination)


class TestLoadRaw:
    def test_columns_load_as_strings(self, tmp_path: Path) -> None:
        """Parsing happens in one explicit place, so nothing may be inferred here.

        Left to pandas, `Zip` would become an integer and lose leading zeros, and
        dates would be inferred inconsistently across chunks.
        """
        path = tmp_path / "f.csv"
        path.write_text("Zip,Term\n01234,84\n", encoding="utf-8")
        frame = ingestion.load_raw(path)
        assert frame["Zip"].iloc[0] == "01234"
        # pandas 3 returns StringDtype rather than object for dtype=str, so the
        # value is asserted rather than the dtype name.
        assert isinstance(frame["Term"].iloc[0], str)
        assert frame["Term"].iloc[0] == "84"

    def test_row_limits_are_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "f.csv"
        path.write_text("a\n" + "".join(f"{i}\n" for i in range(100)), encoding="utf-8")
        assert len(ingestion.load_raw(path, nrows=10)) == 10

    def test_the_full_file_loads_when_no_limit_is_given(self, tmp_path: Path) -> None:
        path = tmp_path / "f.csv"
        path.write_text("a\n" + "".join(f"{i}\n" for i in range(50)), encoding="utf-8")
        assert len(ingestion.load_raw(path)) == 50

    def test_empty_fields_become_null_not_the_string_nan(self, tmp_path: Path) -> None:
        path = tmp_path / "f.csv"
        path.write_text("ChgOffDate,Term\n,84\n", encoding="utf-8")
        assert pd.isna(ingestion.load_raw(path)["ChgOffDate"].iloc[0])


def _fake_urlopen(payload: bytes) -> Any:
    """A urlopen stand-in yielding ``payload`` once, then end of stream."""

    class _Response:
        def __init__(self) -> None:
            self._sent = False

        def read(self, _size: int = -1) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return payload

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def _open(*_args: Any, **_kwargs: Any) -> _Response:
        return _Response()

    return _open
