"""What the tests share.

Where they find what lies outside the package -- the tm5-zoom checkout, the dump sets and the
model's input files -- each an environment variable with a default under the checkout's
scratch tree (the sibling `../tm5-zoom` unless $PYDYN_TM5ZOOM says otherwise); a test that
needs one is skipped without it (`needs`), so read the count, not the colour:

    PYDYN_TM5ZOOM           the tm5-zoom checkout (constants mirror, harness sources)
    PYDYN_SETS              the directory holding the dump sets by name, the default for the
                            next three and the zoom ones (runs/fetch_fixture.py fills one)
    PYDYN_DUMP_DIR          docs/state.md section 3b's physics set (runs/setup_repin.sh)
    PYDYN_FREE_DUMP_DIR     its transport-only set
    PYDYN_CFL_DUMP_DIR      its reduced-time-step set
    PYDYN_METEO_ROOT        the tm5-nc archive those were made from (my.meteo.dir)
    PYDYN_EMIS_DIR          the 1x1 CH4 emission files (emis.ch4.input.dir)
    PYDYN_CHEM_ROOT         the CH4 input tree (ch4loss.inputdir, oh.file, ...)
    PYDYN_IC_FILE           the istart-66 file
    PYDYN_ZOOM_METEO_DIR    four meteo events of zm_replay (section 3f)
    PYDYN_ZOOM_PHYS_DIR     27 events of zp_replay (section 3f)
    PYDYN_SERIAL_DUMP_DIR   the PYD-2 serial set (section 3e)
    PYDYN_MPI_DUMP_DIRS     its 2-, 3- and 5-rank sets, ':'-separated

And the small synthetic configuration the unit tests run on.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pydyn import redgrid, synth
from pydyn.harness import format_report

TM5ZOOM = Path(os.environ.get("PYDYN_TM5ZOOM", Path(__file__).parents[2] / "tm5-zoom"))
SCRATCH = TM5ZOOM / ".tm5-cloud-scratch"


def _env(name, default):
    return Path(os.environ.get(name, default))


SETS = _env("PYDYN_SETS", SCRATCH / "pydyn_build" / "dumps")
DUMP_DIR = _env("PYDYN_DUMP_DIR", SETS / "repin_phys_745e94f70")
FREE_DUMP_DIR = _env("PYDYN_FREE_DUMP_DIR", SETS / "repin_free_745e94f70")
CFL_DUMP_DIR = _env("PYDYN_CFL_DUMP_DIR", SETS / "repin_cfl_745e94f70")
METEO_ROOT = _env("PYDYN_METEO_ROOT", SCRATCH / "METEO" / "tm5-nc")
EMIS_DIR = _env("PYDYN_EMIS_DIR", SCRATCH / "input" / "ch4c13" / "all_source")
CHEM_ROOT = _env("PYDYN_CHEM_ROOT", SCRATCH / "input" / "ch4c13")
IC_FILE = _env("PYDYN_IC_FILE", SCRATCH / "ic" / "ch4_init_glb3x2_mmr.nc")
ZOOM_METEO_DIR = _env("PYDYN_ZOOM_METEO_DIR", SETS / "zm_meteo_fixture")
ZOOM_PHYS_DIR = _env("PYDYN_ZOOM_PHYS_DIR", ZOOM_METEO_DIR.parent / "zp_fixture")
SERIAL_DUMP_DIR = _env("PYDYN_SERIAL_DUMP_DIR", SETS / "pyd2_serial")
_MPI = os.environ.get("PYDYN_MPI_DUMP_DIRS")
MPI_DUMP_DIRS = [Path(d) for d in _MPI.split(":")] if _MPI else [SETS / f"pyd2_mpi{n}" for n in (2, 3, 5)]


@contextmanager
def twins():
    """The reference's libm routed through pydyn.gpu's twin library, then put back."""
    from pydyn import ops
    from pydyn.gpu.backend import host_library

    saved = {name: getattr(ops.libm, name) for name in ("exp", "log", "pow", "cos", "sin", "cube", "path")}
    ops.use_libm(str(host_library()))
    try:
        yield
    finally:
        for name, f in saved.items():
            setattr(ops.libm, name, f if name == "path" else staticmethod(f))


def needs(path, env):
    """Skip unless `path` (named by the variable `env`) exists."""
    return pytest.mark.skipif(not path.exists(), reason=f"no {path} (set {env})")


# ------------------------------------------------------------------ the synthetic configuration
IM, JM, LM, NT = 12, 6, 5, 2
NDYN = 3600
DT = NDYN / 2.0


def small_redgrid():
    return redgrid.RedGrid(IM, JM, jred=(1, 2, JM), clustsize=(IM, 4, 3))


def synth_state(seed=0):
    rng = np.random.default_rng(seed)
    return rng, synth.state(rng, IM, JM, LM, NT)


def assert_bitwise(reports):
    assert all(r["equal"] for r in reports), format_report(reports)
