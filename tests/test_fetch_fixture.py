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
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def record(tmp_path):
    src = tmp_path / "zenodo"
    src.mkdir()
    members = {"setA/e00001_meteo.meta": b"tag = meteo\n", "setA/e00001_m.bin": b"\0" * 8}
    md5 = _archive(src / "setA.tar.gz", members)
    return src, [("setA.tar.gz", (src / "setA.tar.gz").as_uri(), md5)]


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
    (tmp_path / "out" / "setA.tar.gz").write_bytes(b"truncated")
    ff.fetch(files, tmp_path / "out")
    assert (tmp_path / "out" / "setA" / "e00001_m.bin").stat().st_size == 8


@pytest.mark.skipif(not hasattr(tarfile, "data_filter"), reason="no tarfile extraction filter before 3.10.12")
def test_refuses_a_member_outside_dir(tmp_path):
    src = tmp_path / "evil.tar.gz"
    md5 = _archive(src, {"../escaped": b"x"})
    with pytest.raises(tarfile.FilterError):
        ff.fetch([("evil.tar.gz", src.as_uri(), md5)], tmp_path / "out")
    assert not (tmp_path / "escaped").exists()
