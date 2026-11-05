# Security

pydyn is a reference implementation for checking numerical code. It runs on your machine and
calls no network service, except `runs/fetch_fixture.py` (Zenodo, read-only) and the
maintainer's `runs/zenodo_upload.py`.

What it trusts, so you can decide whether to:

- **Libraries it loads.** `ctypes` loads FFTW, LAPACK and libm from the paths in `PYDYN_FFTW`,
  `PYDYN_LAPACK` and `PYDYN_LIBM` (or `--libm`), else from the system's search. Set those only
  to libraries you trust: loading a library runs its code.
- **What it compiles.** A C loop for libm, the emulation kernels and the Metal shim are built
  with `$CC` / `$CXX` / Xcode's tools and cached in `${XDG_CACHE_HOME:-~/.cache}/pydyn`, which
  pydyn uses only while it is owned by you and writable by nobody else.
- **Dump sets and input files.** Read as data: no `pickle`, `eval` or `allow_pickle`. A
  malformed header is refused with the file's name.
- **The fixture.** `runs/fetch_fixture.py` checks each archive's MD5 against the record and, for
  the published record, its SHA-256 against the value pinned in the script, and unpacks only
  plain files inside the target directory.

## Reporting a vulnerability

Please report it privately, through the repository's **Security** tab ("Report a
vulnerability"), rather than in a public issue. You should hear back within two weeks. Fixes
go into the next release; there is one supported version, the latest.
