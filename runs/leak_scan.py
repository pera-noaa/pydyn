#!/usr/bin/env python3
"""Refuse a tree that would leak: the check runs/export_public.sh makes before the public commit.

    python3 runs/leak_scan.py DIR          exit 1 and list every hit, else exit 0

Scans every file git tracks in DIR (text files only).  What it refuses:

  claude      a file named CLAUDE.md / CLAUDE.local.md anywhere, or text citing one
  laptop      a /Users/ path outside runs/MANIFEST.md (whose provenance names the laptop's)
  username    a cluster username in a path: CU identikeys (/projects/abcd1234), NOAA
              First.Last homes, any /home/<name> other than $USER
  token       GitHub, Slack, AWS, OpenAI-style keys, private keys, Bearer values, .netrc
              passwords, and any 40+ character mixed-case alphanumeric run of high entropy
              (Zenodo's tokens have no prefix, so only their shape gives them away)
  email       an address not in EMAILS
  host        a host name not in HOSTS, and private IPv4 addresses

This file and the two that spell out what it looks for (export_public.sh and its test) are
exempt from every check but `token`, which exempts nothing.  Standard library only.
"""

import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

EMAILS = {"ashley.pera@noaa.gov"}
HOSTS = {"github.com", "zenodo.org", "sandbox.zenodo.org", "doi.org", "orcid.org", "pypi.org",
         "files.pythonhosted.org", "www.apache.org", "noaa.gov"}  # fmt: skip
SELF = {"runs/leak_scan.py", "runs/export_public.sh", "tests/test_export_public.py"}
LAPTOP_OK = {"runs/MANIFEST.md"}

TOKENS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[abpors]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"Bearer [A-Za-z0-9._~+/-]{20,}=*"),
    re.compile(r"\bmachine\s+\S+\s+(?:login\s+\S+\s+)?password\s+\S+"),
]
LONG_RUN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{40,}(?![A-Za-z0-9])")
USERNAMES = [
    re.compile(r"/(?:projects|scratch/alpine|scratch/summit|home|glade/u/home|pl/active)/[a-z]{4}[0-9]{4}\b"),
    re.compile(r"/(?:home|work2?/noaa/[a-z0-9_-]+|scratch[12]/[A-Za-z0-9_/-]+)/[A-Z][a-z]+\.[A-Z][A-Za-z]+\b"),
    re.compile(r"/home/(?!\$)[A-Za-z][A-Za-z0-9._-]*"),
]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
TLDS = "gov|edu|org|com|net|io|int|mil|local|internal|lan|corp"
HOST = re.compile(rf"(?<![\w.-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:{TLDS})(?![\w-])")
PRIVATE_IP = re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")
CLAUDE_NAME = re.compile(r"(?:^|/)CLAUDE(?:\.local)?\.md$")


def entropy(s):
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def secret_shaped(run):
    """A long alphanumeric run that mixes cases and digits and is close to random."""
    return (re.search(r"[A-Z]", run) and re.search(r"[a-z]", run) and re.search(r"[0-9]", run)
            and entropy(run) > 4.2)  # fmt: skip


def scan_text(path, text):
    """[(check, line number, match)] for one file's text."""
    hits = []
    exempt = path in SELF
    for n, line in enumerate(text.splitlines(), 1):
        for rx in TOKENS:
            hits += [("token", n, m.group(0)) for m in rx.finditer(line)]
        hits += [("token", n, m.group(0)) for m in LONG_RUN.finditer(line) if secret_shaped(m.group(0))]
        if exempt:
            continue
        if "CLAUDE.md" in line or "CLAUDE.local.md" in line:
            hits.append(("claude", n, "CLAUDE.md"))
        if "/Users/" in line and path not in LAPTOP_OK:
            hits.append(("laptop", n, line.strip()[:80]))
        for rx in USERNAMES:
            hits += [("username", n, m.group(0)) for m in rx.finditer(line)]
        hits += [("email", n, m.group(0)) for m in EMAIL.finditer(line) if m.group(0).lower() not in EMAILS]
        emails = " ".join(m.group(0) for m in EMAIL.finditer(line))
        hits += [("host", n, m.group(0)) for m in HOST.finditer(line)
                 if m.group(0) not in HOSTS and m.group(0) not in emails]  # fmt: skip
        hits += [("host", n, m.group(0)) for m in PRIVATE_IP.finditer(line)]
    return hits


def scan(root):
    root = Path(root)
    files = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    hits = []
    for rel in files.stdout.decode().split("\0"):
        if not rel:
            continue
        if CLAUDE_NAME.search(rel):
            hits.append((rel, "claude", 0, "a CLAUDE file"))
            continue
        data = (root / rel).read_bytes() if (root / rel).is_file() else b""
        if b"\0" in data[:8192]:
            continue
        hits += [(rel, *h) for h in scan_text(rel, data.decode("utf-8", errors="replace"))]
    return hits


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} DIR")
    hits = scan(sys.argv[1])
    for rel, check, n, what in hits:
        print(f"{rel}:{n}: {check}: {what}", file=sys.stderr)
    if hits:
        sys.exit(f"leak_scan: {len(hits)} hits")


if __name__ == "__main__":
    main()
