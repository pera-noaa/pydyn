"""The driver (modelIntegration.F90): do_steps, Proces_Region and tm5.F90's two half steps, on a
state wherever it lives (the bodies through `st.lib`).  Part of pydyn.zoom (see its docstring)."""

from .advect import advect_x_zoom, advect_y_zoom, advect_z_zoom
from .tree import N_OPERATORS

__all__ = [
    "OPERATORS",
    "do_steps",
    "proces_region",
    "step",
]

# ------------------------------------------------------------------ the driver (modelIntegration)
OPERATORS = {"x": advect_x_zoom, "y": advect_y_zoom, "z": advect_z_zoom}


def _record_max(stats, region, axis, st):
    """nloop_max / xi as the Fortran's module diagnostics: a running max per region and axis."""
    if stats is None:
        return
    k = "xyz".index(axis)
    per = stats.setdefault(region, {"nloop_max": [0, 0, 0], "xi": [0.0, 0.0, 0.0]})
    per["nloop_max"][k] = max(per["nloop_max"][k], st["nloop_max"])
    if st["xi"] is not None:  # a device child's z has none
        per["xi"][k] = max(per["xi"][k], st["xi"])


def do_steps(
    zoom, st, region, conv, ndyn, limits=True, inverse="crout", stats=None, sources=None, chemistry=None
):
    """modelIntegration.F90 do_steps: the next operators of splitorderzoom(region), until a
    block boundary -- where put_xedges / put_yedges refill the children unless the operator
    just done was x -- or until a `z` that follows a `y`, so that the children run on the
    parent's xyz before the parent does v, s, c.  `sources` / `chemistry` are optional
    callbacks `f(zoom, st, region)`; left None, `s` and `c` are no-ops, as in a transport-only
    build."""
    lib, order = st.lib, zoom.splitorderzoom[region]
    for _ in range(N_OPERATORS):
        nxt = order[zoom.status[region]]
        if nxt in OPERATORS:
            s = {} if stats is not None else None
            OPERATORS[nxt](zoom, st, region, limits, stats=s)
            _record_max(stats, region, nxt, s)
        elif nxt == "v":
            lib.convec_region(zoom, st, region, conv[region], ndyn, inverse)
        elif nxt == "s":
            if sources is not None:
                sources(zoom, st, region)
        elif nxt == "c":
            if chemistry is not None:
                chemistry(zoom, st, region)
        else:
            raise ValueError(f"do_steps: strange value in splitorderzoom: {nxt!r}")
        zoom.status[region] += 1
        s_now = zoom.status[region]
        if s_now % N_OPERATORS == 0:
            if nxt != "x":
                lib.put_xedges(zoom, st, region)
                lib.put_yedges(zoom, st, region)
            return
        if nxt == "z" and order[s_now - 2] == "y":
            return


def proces_region(zoom, st, region, conv, ndyn, **kw):
    """modelIntegration.F90 Proces_Region: tref_ sub-steps of do_steps, each running the
    children over the same interval after the parent's xyz; then update_parent and m2phlb
    (m2phlb1 for region 1)."""
    for _ in range(zoom.ref(region, "t")):
        do_steps(zoom, st, region, conv, ndyn, **kw)
        for child in zoom.children[region]:
            proces_region(zoom, st, child, conv, ndyn, **kw)
        if zoom.status[region] % N_OPERATORS != 0:
            do_steps(zoom, st, region, conv, ndyn, **kw)
    if region != 1:
        st.lib.update_parent(zoom, st, region)
        st.lib.m2phlb(zoom, st, region)
    else:
        st.lib.m2phlb1(zoom, st, 1)


def step(zoom, st, conv, ndyn, **kw):
    """One ndyn step as tm5.F90 runs it (fluxes already set): every region's status reset,
    then Proces_Region(1) for each of the two half steps."""
    zoom.status = [0] * (zoom.nregions + 1)
    for _ in range(2):
        proces_region(zoom, st, 1, conv, ndyn, **kw)
