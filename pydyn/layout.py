"""Conversions between the TM5 halo layouts and the pydyn layout.

TM5 (tm5_tracer_data / global_data / meteodata, region 1) allocates, in Fortran order with
these bounds:
    m_dat%data                (-1:im+2, -1:jm+2, lm)
    rm_t, rxm_t, rym_t, rzm_t (-1:im+2, -1:jm+2, lm, nt)
    am_t, bm_t, cm_t          (0:im+1, 0:jm+1, 0:lm+1)
and the harness allocates pu (0:im, 1:jm, lm), pv (1:im, 1:jm+1, lm), the convection
inputs (im, jm, lm) and dkg (im, jm, lmax_conv) without halos, i.e. already in the pydyn
shapes.

A dump written with unformatted stream I/O and read back with `np.fromfile(...).reshape(
shape, order='F')` gives arrays whose index 0 is the lower bound above.  The helpers here
slice out the interior in the pydyn convention and put it back.
"""

import numpy as np

__all__ = [
    "DECLARED", "INTERIOR", "bounds", "from_tm5", "interior_section", "read_fortran_stream", "section",
    "tm5_shape", "to_tm5", "write_fortran_stream",
]  # fmt: skip

# The Fortran bounds each array is declared with (tm5_tracer_data, global_data, meteodata; the
# zoom state allocates every region alike) ...
DECLARED = {
    "m": "-1:{im2}, -1:{jm2}, 1:{lm}",
    **dict.fromkeys(("rm", "rxm", "rym", "rzm"), "-1:{im2}, -1:{jm2}, 1:{lm}, 1:{nt}"),
    **dict.fromkeys(("am", "bm", "cm"), "0:{im1}, 0:{jm1}, 0:{lm1}"),
    "pu": "0:{im}, 1:{jm}, 1:{lm}",
    "pv": "1:{im}, 1:{jm1}, 1:{lm}",
    "sp": "-1:{im2}, -1:{jm2}, 1:1",
    "phlb": "-1:{im2}, -1:{jm2}, 1:{lm1}",
    "dxyp": "1:{jm}",
}
# ... and the section of each that is the pydyn-layout interior: the halo cells dropped, the
# boundary faces of the fluxes kept (also the arrays a dump carries beyond the state).
INTERIOR = {
    **dict.fromkeys(("m", "rm", "rxm", "rym", "rzm"), "1:{im}, 1:{jm}, 1:{lm}"),
    "am": "0:{im}, 1:{jm}, 1:{lm}",
    "bm": "1:{im}, 1:{jm1}, 1:{lm}",
    **dict.fromkeys(("cm", "mfw1", "mfw2", "pw1", "pw2"), "1:{im}, 1:{jm}, 0:{lm}"),
    **dict.fromkeys(("pu", "pu1", "pu2", "mfu1", "mfu2"), "0:{im}, 1:{jm}, 1:{lm}"),
    **dict.fromkeys(("pv", "pv1", "pv2", "mfv1", "mfv2"), "1:{im}, 1:{jm1}, 1:{lm}"),
    **dict.fromkeys(("sp", "sp1", "sp2"), "1:{im}, 1:{jm}, 1"),
    "phlb": "1:{im}, 1:{jm}, 1:{lm1}",
    "temper": "1:{im}, 1:{jm}, 1:{lm}",
}


def _fill(text, im, jm, lm, nt=1):
    return text.format(im=im, im1=im + 1, im2=im + 2, jm=jm, jm1=jm + 1, jm2=jm + 2, lm=lm, lm1=lm + 1, nt=nt)


def section(text):
    """'0:60, 1:45, 1' -> (slice(0, 60), slice(1, 45), 1): an inclusive Fortran section."""
    out = []
    for part in text.split(","):
        lo, colon, hi = part.partition(":")
        out.append(slice(int(lo), int(hi)) if colon else int(lo))
    return tuple(out)


def bounds(name, im, jm, lm, nt=1):
    """The declared (lower, upper) bounds of `name`, axis by axis."""
    return [(s.start, s.stop) for s in section(_fill(DECLARED[name], im, jm, lm, nt))]


def interior_section(name, im, jm, lm):
    """The Fortran section of `name` that is its pydyn-layout interior."""
    return section(_fill(INTERIOR[name], im, jm, lm))


def tm5_shape(name, im, jm, lm, nt=1):
    """Full (haloed) shape of a TM5 array as the harness reads/writes it."""
    return tuple(hi - lo + 1 for lo, hi in bounds(name, im, jm, lm, nt))


def _interior_index(name, im, jm, lm):
    """The numpy index of the interior of a full array (index 0 = the declared lower bound)."""
    lower = [lo for lo, _ in bounds(name, im, jm, lm)]
    sec = interior_section(name, im, jm, lm)
    return (*(slice(s.start - lo, s.stop - lo + 1) for s, lo in zip(sec, lower)), Ellipsis)


def from_tm5(name, arr, im, jm, lm):
    """Interior of a TM5 array `name` (one of m, rm, rxm, rym, rzm, am, bm, cm)."""
    return np.ascontiguousarray(arr[_interior_index(name, im, jm, lm)])


def to_tm5(name, arr, im, jm, lm):
    """Embed a pydyn array into a zero-haloed TM5 array of the right bounds."""
    nt = arr.shape[3] if arr.ndim == 4 else 1
    out = np.zeros(tm5_shape(name, im, jm, lm, nt))
    out[_interior_index(name, im, jm, lm)] = arr
    return out


def read_fortran_stream(path, shape, dtype="<f8"):
    """Read one array written by `open(..., access='stream', form='unformatted')`."""
    return np.fromfile(path, dtype=dtype).reshape(shape, order="F")


def write_fortran_stream(path, arr, dtype="<f8"):
    np.asfortranarray(arr, dtype=dtype).T.tofile(path)  # .T of an F-array is C-contiguous
