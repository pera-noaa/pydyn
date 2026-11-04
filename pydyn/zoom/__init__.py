"""Zoom nesting: the region tree and the parent <-> child exchange of TM5-zoom (`PYD-3`).

Transcribed from `tm5-zoom@13149b4a1`, as the production overlay chain resolves it:

    initexit.F90            determine_children_etc, blank_zoom_region, define_splitorderzoom
                            (proj/ct-unified/trunk)                       -> Zoom
    advectx__slopes.F90     advectxzoom, dynamu, compress/uncompress_xedges, put_xedges
    advecty__slopes.F90     advectyzoom, dynamv, compress/uncompress_yedges, put_yedges
                            (proj/ct-unified/trunk)
    advectz__slopes.F90     advectzzoom, dynamw (proj/ct-unified/branches/speedup)
    zoom_tools.F90          mix_edges, update_parent, coarsen_region (proj/ct-unified/trunk)
    advect_tools__speedup   m2phlb, m2phlb1, coarsen_ambm (proj/ct-unified/branches/speedup)
    advect.F90, convection.F90   dynam0 and convec for a child: dt = ndyn / (2 tref), and the
                            child's interior scope isr:ier x jsr:jer
    advectm_cfl.F90         Setup_MassFlow's region loop (proj/ct-unified/trunk)
    modelIntegration.F90    do_steps, Proces_Region (proj/ct-unified/trunk); `step` is the
                            two half steps of tm5.F90's main loop

Unlike the rest of the package, which works on the halo-free pydyn layout (layout.py), the
arrays here keep their Fortran bounds (`FView`): the exchange is halo arithmetic, the
halos of a child carry its parent's boundary cells, and a transcription that renumbered
them would be the first place an index error hides.  `ZoomState.fields[r]` holds
`m (-1:im+2, -1:jm+2, lm)`, `rm..rzm (..., nt)`, `am bm cm (0:im+1, 0:jm+1, 0:lm+1)`,
`pu (0:im, jm, lm)`, `pv (im, jm+1, lm)`, `sp (-1:im+2, -1:jm+2, 1)`,
`phlb (-1:im+2, -1:jm+2, lm+1)` and `dxyp (jm)`, exactly as the model allocates them.
Region 1 is advected by the kernels of kernels.py / redgrid.py on its interior views --
in a zoom configuration its advection differs only by the calls around them -- and a child
by the `_dynam{u,v,w}_child` transcriptions here.  The driver (do_steps, Proces_Region, the
advect*zoom and Check_CFL entry points, coarsen_region's recursion, Setup_MassFlow) runs on a
state wherever it lives: it calls the bodies through `ZoomState.lib`, zoom.host's (these
transcriptions) for NumPy arrays and pydyn.gpu.zoom's for device ones.

Scope.  What production runs: one global region (cyclic, touching both poles, possibly
reduced) and nested children that touch neither pole, are not cyclic, and sit strictly
inside their parent in x (`1 < ibeg`, `iend < im(parent)`).  Anything else raises
`NotImplementedError` at construction, each for a reason visible in the Fortran:
pole-touching and cyclic children take branches of dynamv/mix_edges that no configuration
uses; a child at the parent's first or last column has put_xedges read the parent's halo
(`ibeg-1 = 0`), which is only as fresh as the parent's last x step; and a child across the
dateline takes update_parent's `ibeg > iend` branch, which indexes the parent with the
*child's* `im`.
"""

# the kernels whose sub-cycle caps tests/test_constants.py reads off their signatures
from .advect import (  # noqa: F401
    _dynamu_child,
    _dynamu_region1,
    _dynamv_child,
    _dynamw_child,
    advect_x_zoom,
    advect_y_zoom,
    advect_z_zoom,
)
from .check_cfl import (  # noqa: F401
    CFL_N_OPERATORS,
    CFL_SPLITORDER,
    _advectm_x,
    _advectm_y,
    _advectm_z,
    cfl_splitorderzoom,
    check_cfl_zoom,
    determine_cfl_iter,
)
from .exchange import mix_edges, put_xedges, put_yedges, update_parent
from .massflow import coarsen_ambm, coarsen_region, dynam0, m2phlb, m2phlb1, setup_massflow
from .physics import chemistry_region, convec_region, owned, scope, sources_region
from .state import FIELDS, FView, ZoomState
from .steps import OPERATORS, do_steps, proces_region, step
from .tree import N_OPERATORS, SPLITORDER, Region, Zoom

__all__ = [
    "CFL_N_OPERATORS",
    "CFL_SPLITORDER",
    "FIELDS",
    "N_OPERATORS",
    "OPERATORS",
    "SPLITORDER",
    "FView",
    "Region",
    "Zoom",
    "ZoomState",
    "advect_x_zoom",
    "advect_y_zoom",
    "advect_z_zoom",
    "cfl_splitorderzoom",
    "check_cfl_zoom",
    "chemistry_region",
    "coarsen_ambm",
    "coarsen_region",
    "convec_region",
    "determine_cfl_iter",
    "do_steps",
    "dynam0",
    "m2phlb",
    "m2phlb1",
    "mix_edges",
    "owned",
    "proces_region",
    "put_xedges",
    "put_yedges",
    "scope",
    "setup_massflow",
    "sources_region",
    "step",
    "update_parent",
]
