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
    """The conditions every result in the run holds under: NumPy's version and SIMD paths."""
    import numpy as np

    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    except ImportError:  # NumPy 1.x
        from numpy.core._multiarray_umath import __cpu_dispatch__, __cpu_features__

    dispatched = [f for f in __cpu_dispatch__ if __cpu_features__.get(f)]
    return (f"numpy {np.__version__} on {platform.machine()}, SIMD dispatched: "
            f"{' '.join(dispatched) or 'none'}")
