"""The bodies the driver (steps, the advect*zoom and advectm* entry points, coarsen_region,
setup_massflow, check_cfl_zoom) runs through `ZoomState.lib` for a state in host memory.
pydyn.gpu.zoom has the same names for a state on a device.  Part of pydyn.zoom (see its
docstring)."""

from .advect import (  # noqa: F401
    _dynamu_child,
    _dynamu_region1,
    _dynamv_child,
    _dynamv_region1,
    _dynamw_child,
    _dynamw_region1,
    outer_x_faces_zeroed,
    outer_y_faces_zeroed,
)
from .check_cfl import (  # noqa: F401
    _dynamum_child,
    _dynamum_region1,
    _dynamvm_child,
    _dynamvm_region1,
    _dynamwm,
)
from .exchange import (  # noqa: F401
    _uncompress_xedges,
    _uncompress_yedges,
    mix_edges,
    put_xedges,
    put_yedges,
    update_parent,
)
from .massflow import _coarsen_block, coarsen_ambm, dynam0, m2phlb, m2phlb1  # noqa: F401
from .physics import chemistry_region, convec_region, sources_region  # noqa: F401

__all__ = [
    "BODIES",
    "assign",
    "restore",
    "save",
    "scale",
]

BODIES = (
    "_coarsen_block _dynamu_child _dynamu_region1 _dynamum_child _dynamum_region1 _dynamv_child "
    "_dynamv_region1 _dynamvm_child _dynamvm_region1 _dynamw_child _dynamw_region1 _dynamwm "
    "_uncompress_xedges _uncompress_yedges assign chemistry_region coarsen_ambm convec_region dynam0 "
    "m2phlb m2phlb1 mix_edges outer_x_faces_zeroed outer_y_faces_zeroed put_xedges put_yedges "
    "restore save scale sources_region update_parent"
).split()


def assign(dst, src, region, names):
    """dst's named arrays of the region set to src's."""
    for k in names:
        dst[region][k].data[...] = src[region][k].data


def save(st, key, names):
    """Copies of the named arrays of every region, for restore(st, key)."""
    st.saved[key] = {(r, k): f[k].data.copy() for r, f in st.fields.items() for k in names}


def restore(st, key):
    for (r, k), v in st.saved[key].items():
        st[r][k].data[...] = v


def scale(st, names, factor):
    """a[...] = a * factor for the named arrays of every region."""
    for f in st.fields.values():
        for k in names:
            f[k].data[...] = f[k].data * factor
