"""Make `import pydyn` work when pytest is run from the repository root, and pin NumPy's
SIMD dispatch before anything imports it.

On x86-64 with AVX-512, NumPy dispatches the vectorised `log`, `exp`, `cbrt` and `power` to
SVML-derived kernels that are not the C library's.  The bitwise tier never sees them (the
transcription's transcendentals go through `pydyn.ops.libm`, scalar), but the simple-vs-fast
tier does: `simple_diffusion` calls `np.log` where `diffusion` calls `libm.log`, and on Orion
(Xeon Gold 6148, NumPy 2.0.2) a 1-ulp difference in `ustar` grows to 782 ulp in `dkg` at
seeds 18-20 of `test_calc_kzz`.  With AVX-512 dispatch off the same three pass within the
16-ulp bound, so the difference is the dispatch, not the NumPy version (`PYD-8`).  Turning it
off here makes the simple tier measure the transcription and nothing else.  NumPy 2.4 renamed
the x86 targets to microarchitecture levels (X86_V3, X86_V4), so the names depend on the
version, and disabling X86_V4 leaves AVX512_SPR on where a wheel dispatches it (NumPy 2.5.4's
manylinux cp314 on a Xeon 6975P-C); a name NumPy does not dispatch only draws an ImportWarning.
Set NPY_DISABLE_CPU_FEATURES yourself (even to "") to override.
"""

import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

if platform.machine() in ("x86_64", "AMD64"):
    _numpy = tuple(int(p) for p in version("numpy").split(".")[:2])
    os.environ.setdefault(
        "NPY_DISABLE_CPU_FEATURES",
        "X86_V4 AVX512_ICL AVX512_SPR" if _numpy >= (2, 4)
        else "AVX512F AVX512CD AVX512_KNL AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR",
    )

sys.path.insert(0, str(Path(__file__).parent))


def pytest_report_header(config):
    """The conditions every result in the run holds under: NumPy's version and SIMD paths, and
    the tm5-zoom commits the Fortran-side tests read."""
    import numpy as np

    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    except ImportError:  # NumPy 1.x
        from numpy.core._multiarray_umath import __cpu_dispatch__, __cpu_features__

    dispatched = [f for f in __cpu_dispatch__ if __cpu_features__.get(f)]
    return [f"numpy {np.__version__} on {platform.machine()}, SIMD dispatched: "
            f"{' '.join(dispatched) or 'none'}", *_tm5zoom_header()]


def _tm5zoom_header():
    """The tm5-zoom checkout's commit (test_constants reads it) and each built harness's
    (fortran/<binary>.tm5zoom), against fortran/TM5ZOOM_PIN, the commit the claims are made at."""
    import subprocess

    fortran = Path(__file__).parent / "fortran"
    if not (fortran / "TM5ZOOM_PIN").is_file():  # an sdist
        return []
    pin = (fortran / "TM5ZOOM_PIN").read_text().strip()
    root = Path(os.environ.get("PYDYN_TM5ZOOM", Path(__file__).parents[1] / "tm5-zoom"))
    at = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    at = at.stdout.strip() if at.returncode == 0 else ""
    state = ("absent" if not root.is_dir() else "at the pin" if at == pin
             else f"at {at[:9] or 'no git commit'}, NOT the pin")
    lines = [f"tm5-zoom pin {pin[:9]}; checkout {root}: {state}"]
    for name in ("harness", "harness_zoom2", "harness_zoom3"):
        stamp = fortran / f"{name}.tm5zoom"
        if stamp.is_file():
            built = stamp.read_text().strip()
            lines.append(f"fortran/{name}: built at the pin" if built == pin
                         else f"fortran/{name}: built from {built}, NOT the pin")
        elif (fortran / name).exists():
            lines.append(f"fortran/{name}: no .tm5zoom stamp, commit unknown (rebuild it)")
    return lines
