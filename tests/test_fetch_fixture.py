"""runs/fetch_fixture.py takes an archive only at the record's md5, and unpacks only into DIR.

Offline: the listing is built here with file:// URLs, which urlopen reads like the record's.
"""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "runs" / "fetch_fixture.py"
spec = importlib.util.spec_from_file_location("fetch_fixture", SCRIPT)
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)


def _archive(path, members):
    with tarfile.open(path, f"w:{path.suffix[1:]}") as tar:  # .xz as the fixture, .gz still read
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(params=["xz", "gz"])
def record(tmp_path, request):
    src = tmp_path / "zenodo"
    src.mkdir()
    name = f"setA.tar.{request.param}"
    members = {"setA/e00001_meteo.meta": b"tag = meteo\n", "setA/e00001_m.bin": b"\0" * 8}
    md5 = _archive(src / name, members)
    return src, [(name, (src / name).as_uri(), md5)]


def test_unpacks_at_the_records_md5(tmp_path, record):
    _, files = record
    ff.fetch(files, tmp_path / "out")
    assert (tmp_path / "out" / "setA" / "e00001_meteo.meta").read_bytes() == b"tag = meteo\n"


def test_refuses_a_corrupted_download(tmp_path, record):
    (name, url, _), = record[1]
    with pytest.raises(SystemExit, match="md5"):
        ff.fetch([(name, url, "0" * 32)], tmp_path / "out")
    assert not (tmp_path / "out" / "setA").exists()


def test_an_archive_at_the_md5_is_not_downloaded_again(tmp_path, record, monkeypatch):
    _, files = record
    ff.fetch(files, tmp_path / "out")
    monkeypatch.setattr(ff, "download", lambda *a: pytest.fail("downloaded again"))
    ff.fetch(files, tmp_path / "out")


def test_an_unpacked_archive_is_neither_hashed_nor_unpacked_again(tmp_path, record, monkeypatch):
    _, files = record
    ff.fetch(files, tmp_path / "out")
    monkeypatch.setattr(ff, "md5", lambda *a: pytest.fail("hashed again"))
    monkeypatch.setattr(ff.tarfile, "open", lambda *a: pytest.fail("unpacked again"))
    ff.fetch(files, tmp_path / "out")


def test_a_changed_archive_is_checked_again(tmp_path, record):
    _, files = record
    ff.fetch(files, tmp_path / "out")
    (tmp_path / "out" / files[0][0]).write_bytes(b"truncated")  # size and mtime no longer the stamp's
    (tmp_path / "out" / "setA" / "e00001_m.bin").unlink()
    ff.fetch(files, tmp_path / "out")
    assert (tmp_path / "out" / "setA" / "e00001_m.bin").stat().st_size == 8


def test_an_unpublished_record_is_a_message(monkeypatch):
    def refuse(url, timeout):
        raise ff.urllib.error.HTTPError(url, 404, "NOT FOUND", {}, None)

    monkeypatch.setattr(ff.urllib.request, "urlopen", refuse)
    with pytest.raises(SystemExit, match="not published yet"):
        ff.listing("1")


def test_a_wrong_archive_in_place_is_replaced(tmp_path, record):
    _, files = record
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / files[0][0]).write_bytes(b"truncated")
    ff.fetch(files, tmp_path / "out")
    assert (tmp_path / "out" / "setA" / "e00001_m.bin").stat().st_size == 8


def test_nothing_is_unpacked_until_every_md5_checks(tmp_path, record):
    src, files = record
    other = src / "setB.tar.xz"
    _archive(other, {"setB/e00001_m.bin": b"\0" * 8})
    with pytest.raises(SystemExit, match=r"setB\.tar\.xz: md5"):
        ff.fetch([*files, (other.name, other.as_uri(), "0" * 32)], tmp_path / "out")
    assert not (tmp_path / "out" / "setA").exists()


@pytest.mark.parametrize("members", [
    {"../escaped": b"x"},
    {"/abs/escaped": b"x"},
])
def test_refuses_a_member_outside_dir(tmp_path, members):
    """On every Python: the members are checked before tarfile's own filter (absent < 3.10.12)."""
    src = tmp_path / "evil.tar.xz"
    md5 = _archive(src, members)
    with pytest.raises(SystemExit, match="refusing the member"):
        ff.fetch([(src.name, src.as_uri(), md5)], tmp_path / "out")
    assert not (tmp_path / "escaped").exists() and not Path("/abs/escaped").exists()


def test_refuses_a_link_member(tmp_path):
    src = tmp_path / "link.tar.xz"
    with tarfile.open(src, "w:xz") as tar:
        info = tarfile.TarInfo("setA/passwd")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(info)
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="refusing the member"):
        ff.fetch([(src.name, src.as_uri(), md5)], tmp_path / "out")


def _json(obj):
    return io.BytesIO(json.dumps(obj).encode())


@pytest.mark.parametrize("name", ["../x.tar.xz", "/etc/x.tar.xz", ".hidden.tar.xz", "a/b.tar.xz"])
def test_refuses_a_file_name_that_is_a_path(monkeypatch, name):
    listing = {"files": [{"key": name, "links": {"self": "file:///x"}, "checksum": "md5:0"}]}
    monkeypatch.setattr(ff.urllib.request, "urlopen", lambda url, timeout: _json(listing))
    with pytest.raises(SystemExit, match="refusing the file name"):
        ff.listing("1")


def test_the_published_record_must_be_the_pinned_files(monkeypatch):
    listing = {"files": [{"key": "other.tar.xz", "links": {"self": "file:///x"}, "checksum": "md5:0"}]}
    monkeypatch.setattr(ff.urllib.request, "urlopen", lambda url, timeout: _json(listing))
    with pytest.raises(SystemExit, match="not the pinned ones"):
        ff.listing(ff.RECORD)


def test_a_pinned_archive_at_another_sha256_is_refused(tmp_path, monkeypatch):
    """The record's md5 agrees (it came with the listing) but the pinned SHA-256 does not."""
    name = sorted(ff.SHA256)[0]
    src = tmp_path / "zenodo"
    src.mkdir()
    md5 = _archive(src / name, {"setA/e00001_m.bin": b"\0" * 8})
    with pytest.raises(SystemExit, match="SHA-256"):
        ff.fetch([(name, (src / name).as_uri(), md5)], tmp_path / "out")
    assert not (tmp_path / "out" / "setA").exists()
