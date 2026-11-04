"""convec, do_add_2d and ch4_chemistry_step on the cells a region owns.  Part of pydyn.zoom (see
its docstring)."""

import numpy as np

from ..kernels import (
    convec,
)


def scope(zoom, region):
    """The interior scope isr:ier x jsr:jer as 0-based slices of an (im, jm) array."""
    return (slice(zoom.isr[region] - 1, zoom.ier[region]), slice(zoom.jsr[region] - 1, zoom.jer[region]))


def owned(zoom, region):
    """(im, jm) bool: the cells of the scope that the region owns (`zoomed == region`) -- where
    convec, do_add_2d and ch4_chemistry_step act; the rest is a child's or an edge."""
    g = zoom.g(region)
    own = np.zeros((g.im, g.jm), dtype=bool)
    own[scope(zoom, region)] = True
    return own & (zoom.zoomed[region] == region)


def convec_region(zoom, st, region, conv, ndyn, inverse="crout"):
    """convection.F90 convec for any region: the columns of the interior scope isr:ier x
    jsr:jer that the region owns (`zoomed == region`), dt = nconv / (2 tref), nconv = ndyn.
    `conv`: entu detu entd detd dkg (im, jm, >= lmax) and cloud_top cloud_lfs (im, jm), in
    the kernels.convec layout, plus 'lmax'."""
    g, f = zoom.g(region), st[region]
    im, jm, lm = g.im, g.jm, zoom.lm
    cols = np.nonzero(owned(zoom, region))

    def gather(q):  # (im, jm, ...) -> (ncol, 1, ...)
        return q[cols][:, None]

    fields = [f[name][1:im, 1:jm, 1:lm] for name in ("m", "rm", "rxm", "rym", "rzm")]
    work = [gather(q).copy() for q in fields]
    names = ("entu", "detu", "entd", "detd", "dkg", "cloud_top", "cloud_lfs")
    meteo = {k: gather(np.asarray(conv[k])) for k in names}
    dt = ndyn / (2.0 * g.tref)
    convec(*work, **meteo, dt=dt, lmax=conv["lmax"], inverse=inverse)
    for q, w in zip(fields, work, strict=True):
        q[cols] = w[:, 0]


def sources_region(zoom, st, region, injections):
    """emission_data.F90 do_add_2d for any region: each `injections` item (x (im, jm) in kg,
    l_level, tracer, both 1-based) added to rm and subtracted from rzm on the owned cells."""
    g, f = zoom.g(region), st[region]
    own = owned(zoom, region)
    for x, level, tracer in injections:
        rm = f["rm"][1 : g.im, 1 : g.jm, level, tracer]
        rzm = f["rzm"][1 : g.im, 1 : g.jm, level, tracer]
        rm[own] = rm[own] + x[own]
        rzm[own] = rzm[own] - x[own]


def chemistry_region(zoom, st, region, decay, tracers):
    """chemistry_ch4.F90 ch4_chemistry_step's phase 2 for any region: rm and the slopes of the
    (1-based) `tracers` scaled by `decay` (the scope's, chem.decay of its loss rates) on the
    owned cells; the rest keep factor 1.0 (with_zoom), which leaves them as they are."""
    g, f = zoom.g(region), st[region]
    factor = np.ones((g.im, g.jm, zoom.lm))
    factor[scope(zoom, region)] = decay
    factor[~owned(zoom, region)] = 1.0
    for t in tracers:
        for name in ("rm", "rxm", "rym", "rzm"):
            q = f[name][1 : g.im, 1 : g.jm, 1 : zoom.lm, t]
            q[...] = q * factor
