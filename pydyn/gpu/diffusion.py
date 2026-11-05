"""diffusion.calc_kzz on the device (csrc/kzz.cuh): the same arguments, dkg back as a device
array.  The constants are diffusion.py's own, handed to the kernel in one array; the zonal
cell widths dxx are _winds' expression, evaluated on the host with ops.libm's cos."""

import numpy as np

from .. import diffusion as D
from ..ops import libm
from .backend import backend

__all__ = ["calc_kzz"]

_CONSTS = (
    "GRAV",
    "VKARMAN",
    "RHO_AIR",
    "CP_AIR",
    "LVAP",
    "CKMIN",
    "SFFRAC",
    "ONET",
    "APZERO",
    "CCON",
    "BETAH",
    "BINM",
    "FAK",
    "FAKN",
    "ZKAPPA",
    "ZCRDQ",
    "TINY",
    "ZACB",
    "RICR",
    "SSFRAC",
    "KVHMIN",
)


def calc_kzz(spm, sp, pu, pv, t, q, surface, at, bt, dxyp, lmax_conv, dx=6.0, dy=4.0, ybeg=-90.0, pblh=None):
    """dkg (im, jm, lmax_conv) of one step on the device; `pblh` (a device (im, jm) array) is
    filled with the boundary-layer height when given."""
    be = backend()
    xp = be.xp

    def dev(x):
        return be.contiguous(x) if be.is_device(x) else be.to_device(np.asarray(x, dtype=float), "input")

    im, jm, lm = t.shape
    from .kernels import _check_lmk

    lmk = _check_lmk(lm)
    jm_ = np.arange(jm)
    lat = ybeg + 0.5 * dy + dy * jm_
    dxx = D.AE * dx * D.GTOR * libm.cos(lat * D.GTOR)
    dyy = D.AE * dy * D.GTOR
    k = be.to_device(np.array([getattr(D, name) for name in _CONSTS] + [dyy], dtype=float), "input")
    surf = be.to_device(
        np.stack(
            [
                np.asarray(surface[n], dtype=float)
                for n in ("oro", "lsm", "sr", "srols", "u10m", "v10m", "sshf", "slhf", "ewss", "nsss")
            ]
        ),
        "input",
    )
    pu, pv, t, q = dev(pu), dev(pv), dev(t), dev(q)
    out = xp.zeros((im, jm, lmax_conv))
    be.launch(
        "pd_calc_kzz",
        im * jm,
        im,
        jm,
        lm,
        lmax_conv,
        k,
        dev(spm),
        dev(sp),
        pu,
        *be.strides(pu),
        pv,
        *be.strides(pv),
        t,
        q,
        *be.strides(t),
        surf,
        dev(at),
        dev(bt),
        dev(dxyp),
        dev(dxx),
        out,
        pblh,
        defines={"PD_LMK": lmk},
    )
    return out
