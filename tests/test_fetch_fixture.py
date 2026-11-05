"""runs/fetch_fixture.py takes an archive only at the record's md5, and unpacks only into DIR.

Offline: the listing is built here with file:// URLs, which urlretrieve reads like the record's.
"""

import hashlib
import importlib.util
import io
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
    monkeypatch.setattr(ff.urllib.request, "urlretrieve", lambda *a: pytest.fail("downloaded again"))
    ff.fetch(files, tmp_path / "out")


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


@pytest.mark.skipif(not hasattr(tarfile, "data_filter"), reason="no tarfile extraction filter before 3.10.12")
def test_refuses_a_member_outside_dir(tmp_path):
    src = tmp_path / "evil.tar.xz"
    md5 = _archive(src, {"../escaped": b"x"})
    with pytest.raises(tarfile.FilterError):
        ff.fetch([(src.name, src.as_uri(), md5)], tmp_path / "out")
    assert not (tmp_path / "escaped").exists()

