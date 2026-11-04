"""Every region's state, its arrays with their Fortran bounds (layout.DECLARED).  Part of
pydyn.zoom (see its docstring)."""

from ..dump import FView
from ..layout import bounds

# ------------------------------------------------------------------ the state
FIELDS = ("m", "rm", "rxm", "rym", "rzm", "am", "bm", "cm", "pu", "pv", "sp", "phlb", "dxyp")


class ZoomState:
    """Every region's arrays with their Fortran bounds: `fields[r][name]` is an FView.  `lib`
    holds the bodies the driver runs on them: zoom.host for NumPy arrays, pydyn.gpu.zoom for
    device ones."""

    def __init__(self, fields, lib=None):
        self.fields = fields
        if lib is None:
            from . import host as lib
        self.lib = lib
        self.saved = {}  # host.save's copies

    @classmethod
    def zeros(cls, zoom, nt):
        lm = zoom.lm
        fields = {}
        for r in zoom.regions():
            im, jm = zoom.g(r).im, zoom.g(r).jm
            fields[r] = {name: FView.zeros(*bounds(name, im, jm, lm, nt)) for name in FIELDS}
        return cls(fields)

    def __getitem__(self, r):
        return self.fields[r]

    def copy(self):
        return ZoomState({r: {k: v.copy() for k, v in f.items()} for r, f in self.fields.items()}, self.lib)

    @property
    def nt(self):
        return self.fields[1]["rm"].data.shape[3]


def _interior(f, im, jm, lm):
    """Region-1 views in the pydyn layout (layout.py) of the halo arrays -- writes go through."""
    return (
        f["m"][1:im, 1:jm, 1:lm],
        *(f[q][1:im, 1:jm, 1:lm] for q in ("rm", "rxm", "rym", "rzm")),
    )
