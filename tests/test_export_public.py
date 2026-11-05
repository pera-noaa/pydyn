"""runs/export_public.sh on throwaway repos: what it exports, and that each refusal fires.
The strings it must refuse are built at run time, so this file holds none of them whole."""

import os
import random
import shutil
import string
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "runs" / "export_public.sh"
SCAN = SCRIPT.with_name("leak_scan.py")
pytestmark = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("git"),
                                reason="needs bash and git")


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout


def commit_all(repo, msg="c"):
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg)


@pytest.fixture
def repos(tmp_path):
    """A lab repo holding the script, and a public clone of a bare origin on main."""
    lab, origin, pub = tmp_path / "lab", tmp_path / "origin.git", tmp_path / "pub"
    (lab / "runs").mkdir(parents=True)
    shutil.copy(SCRIPT, lab / "runs" / "export_public.sh")
    shutil.copy(SCAN, lab / "runs" / "leak_scan.py")
    (lab / "README.md").write_text("pydyn\n")
    (lab / "CLAUDE.md").write_text("lab notes\n")
    (lab / "sub").mkdir()
    (lab / "sub" / "CLAUDE.md").write_text("nested lab notes\n")
    (lab / "runs" / "MANIFEST.md").write_text("made in $HOME/tm5-zoom\n")
    git(tmp_path, "init", "-q", "-b", "master", str(lab))
    commit_all(lab)
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(pub))
    git(pub, "checkout", "-q", "-b", "main")
    (pub / "old.txt").write_text("deleted in the lab\n")
    commit_all(pub)
    git(pub, "push", "-q", "origin", "main")
    return lab, pub


def run(lab, pub, *args, remote="origin.git"):
    env = {**os.environ, "PYDYN_PUBLIC_REMOTE": remote}
    return subprocess.run(["bash", str(lab / "runs" / "export_public.sh"), str(pub), *args],
                          capture_output=True, text=True, env=env)


def _token():
    rng = random.Random(5)
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(60))


def test_exports_head_without_claude_md_and_drops_deleted_files(repos):
    lab, pub = repos
    r = run(lab, pub, "-m", "release")
    assert r.returncode == 0, r.stderr
    files = set(git(pub, "ls-files").split())
    assert files == {"README.md", "runs/MANIFEST.md", "runs/export_public.sh", "runs/leak_scan.py"}
    assert git(pub, "log", "-1", "--format=%an <%ae> %s").strip() == \
        "Ash Pera <ashley.pera@noaa.gov> release"
    assert run(lab, pub).returncode == 0  # a second export finds nothing to do
    assert "nothing to export" in run(lab, pub).stdout


def test_without_a_message_stages_only(repos):
    lab, pub = repos
    head = git(pub, "rev-parse", "HEAD")
    assert run(lab, pub).returncode == 0
    assert git(pub, "rev-parse", "HEAD") == head
    assert "README.md" in git(pub, "diff", "--cached", "--name-only")


@pytest.mark.parametrize("path, text, why", [
    ("docs.md", "see CLAUDE" + ".md for the rules\n", "claude"),
    ("notes.md", "data in /Us" + "ers/someone/scratch\n", "laptop"),
    ("runs/MANIFEST.md", "made in /Us" + "ers/someone/tm5-zoom\n", "laptop"),
    ("key.txt", "gh" + "p_" + "a" * 36 + "\n", "token"),
    ("zenodo.txt", "token = " + _token() + "\n", "token"),
    ("aws.txt", "AK" + "IA" + "ABCDEFGHIJKLMNOP\n", "token"),
    ("job.slurm", "cd /projects/" + "abcd1234/pydyn\n", "username"),
    ("job2.slurm", "cd /home/" + "Ash.Pera/pydyn\n", "username"),
    ("who.md", "mail bob" + "@example.com\n", "email"),
    ("net.md", "ssh node3.cluster" + ".internal or 10.0" + ".4.17\n", "host"),
])
def test_refuses_and_resets(repos, path, text, why):
    lab, pub = repos
    (lab / path).write_text(text)
    commit_all(lab)
    head = git(pub, "rev-parse", "HEAD")
    r = run(lab, pub, "-m", "release")
    assert r.returncode != 0 and why in r.stderr
    assert git(pub, "rev-parse", "HEAD") == head
    assert git(pub, "status", "--porcelain") == ""


def test_refuses_uncommitted_lab_changes(repos):
    lab, pub = repos
    (lab / "README.md").write_text("edited\n")
    r = run(lab, pub)
    assert r.returncode != 0 and "uncommitted" in r.stderr


def test_refuses_a_public_clone_behind_origin(repos, tmp_path):
    lab, pub = repos
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", "-b", "main", str(tmp_path / "origin.git"), str(other))
    (other / "x.txt").write_text("x\n")
    commit_all(other)
    git(other, "push", "-q", "origin", "main")
    r = run(lab, pub)
    assert r.returncode != 0 and "behind origin/main" in r.stderr


@pytest.mark.parametrize("arg", ["", "missing"])
def test_refuses_an_empty_or_missing_clone_argument(repos, arg, tmp_path):
    lab, _ = repos
    r = subprocess.run(["bash", str(lab / "runs" / "export_public.sh"), arg], capture_output=True,
                       text=True, cwd=tmp_path)
    assert r.returncode != 0 and ("empty" in r.stderr or "not a directory" in r.stderr)


def test_refuses_the_lab_itself(repos):
    lab, _ = repos
    r = run(lab, lab)
    assert r.returncode != 0 and "is this repo" in r.stderr


def test_refuses_a_clone_of_another_remote(repos):
    lab, pub = repos
    r = run(lab, pub, remote="pera-noaa/pydyn")
    assert r.returncode != 0 and "not pera-noaa/pydyn" in r.stderr
    assert git(pub, "status", "--porcelain") == ""
