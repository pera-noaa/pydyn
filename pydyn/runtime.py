"""The libraries a run of the reference calls into, and the command-line options the replays share.

Bitwise needs the model's own libraries (docs/state.md section 3c): its FFT for the Poisson
balance (`dfftw_*`; MKL's FFTW interface under ifort), its LAPACK for convection's
DGETRF/DGETRI, and its libm for exp/log/pow/cos/sin and ifort's integer power (libimf).  Each
is named on the command line or by its variable -- $PYDYN_FFTW, $PYDYN_LAPACK, $PYDYN_LIBM
(the last applied when `ops` is imported) -- else searched for (balance.fftw_library,
lapack.lapack_library); `describe` says which were taken, for the run's log.

`--lapack twin` and `--libm twin` name pydyn.gpu's host library: the open DGETRF/DGETRI and
exp/log/pow its device kernels run, so that a CPU run and a device run compute the same bits
(pydyn.gpu.check).  `--lapack crout` is the build without with_lapack (kernels.fastminv).
"""

from . import balance, lapack, ops

__all__ = ["add_input_args", "add_library_args", "apply", "describe", "fft_choice", "lapack_choice"]


def fft_choice(fft):
    """--fft auto | fftw | numpy | <path> as balance.poisson_solver takes it."""
    return {"fftw": True, "numpy": False}.get(fft, fft)


def lapack_choice(choice):
    """--lapack auto | lapack | numpy | crout | twin | <path> as lapack.batched_inverse takes it."""
    if choice == "twin":
        return str(_twin())
    return {"lapack": True}.get(choice, choice)


def _twin():
    from .gpu.backend import host_library

    return host_library()


def add_library_args(p):
    p.add_argument("--fft", default="auto", help="auto | fftw | numpy | path of libfftw3 (default: auto)")
    p.add_argument("--lapack", default="auto",
                   help="auto | lapack | numpy | crout | twin | path of a LAPACK library")
    p.add_argument("--libm", metavar="PATH",
                   help="exp/log/pow/cos/sin from this library (ifort: libimf.so; twin)")


def add_input_args(p):
    p.add_argument("--meteo", metavar="ROOT", help="the meteo from the tm5-nc archive under ROOT")
    p.add_argument("--emissions", metavar="DIR", help="the CH4 emission from the 1x1 monthly files in DIR")
    p.add_argument("--chemistry", metavar="ROOT", help="the loss-rate fields from the CH4 input tree ROOT")
    p.add_argument("--initial", metavar="FILE", help="the initial tracer field from the istart-66 FILE")


def apply(args):
    """Take the libm the command line names (the FFT and LAPACK are passed per call)."""
    if args.libm:
        ops.use_libm(str(_twin()) if args.libm == "twin" else args.libm)


def describe(fft="auto", choice="auto"):
    """One line naming the FFT, LAPACK and libm a run with these choices calls."""
    fft, choice = fft_choice(fft), lapack_choice(choice)
    f = "numpy" if fft is False else (balance.fftw_library() if fft in ("auto", True) else fft) or "numpy"
    lp = "numpy" if choice in ("numpy", False) else (
        lapack.lapack_library() if choice in ("auto", True) else choice) or "numpy"  # fmt: skip
    return f"fft {f}  lapack {lp}  libm {ops.libm.path or 'the C library'}"
