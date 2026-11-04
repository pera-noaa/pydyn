"""The `c` operator: Chemistry_Step -> ch4_chemistry_step, CH4-only path
(proj/ct-unified/branches/chem_unified/src/chemistry_ch4.F90).  First-order decay of every
CH4 tracer by four loss reactions whose rates ch4_loss_kernel builds per column:

    1  OH      Spivakovsky 3-D OH below the tropopause: 2.45e-12 exp(-1775/T) OH(i,j,l) oh_scale;
               above it the Bruehl zonal loss-rate column rebinned onto the model levels
    2  Cl      the Bruehl zonal Cl column, plus below the tropopause the Hossaini tropospheric
               Cl rebinned onto the model levels: 2.36e-12 (T/298)^1.37 exp(-939/T) Cl cl_scale
    3  O(1D)   the Bruehl zonal O(1D) column, o1d_scale
    4  soil    -soil_rate * soil(i,j) in the lowest level only (a source, hence the sign)

    tropopause(j) = 30000 - 21500 cos^2(lat)     Pa, the level counts where phlb(l) < it
    decay = exp(-(((L1 + L2) + L3) + L4) dt) ;   rm, rxm, rym, rzm *= decay

`rebin` is mod_emission_regrid.rebin_levels with ip_mix = .true. (a thickness-weighted mean
over the overlap, in descending pressure), written for whole arrays: each output layer sums,
from zero and in order, the input layers that overlap it -- the ones the Fortran skips or
exits on contribute nothing; `rebin_many` works the overlaps out once for several columns.
The zonal columns are given on pressures that are made into edges by
`strat_edges` (midpoints, a half layer below the first, zero above the last, hPa -> Pa).
Cells the rebin does not reach (the model edge below the first input edge) keep the first
input value.

exp, pow and cos go through libm value by value (`ops.libm`), which is what the Fortran
calls; NumPy's own vectorised exp is not guaranteed to round the same way.  Which libm is
part of the claim: gfortran calls the system one, ifort its own libimf, 2-3 ulp apart on
these rates (`ops.use_libm`, `replay --libm`; docs/state.md 3c).
"""

import numpy as np

from .ops import libm, seqsum

__all__ = [
    "CL_A", "CL_EA", "CL_N", "CL_T0", "OH_A", "OH_EA", "SETTINGS", "TROP_DP", "TROP_P0",
    "apply_decay", "decay", "libm", "loss_inputs", "loss_rates", "rebin", "rebin_many",
    "strat_edges", "tropopause",
]  # fmt: skip

# ch4_loss_kernel writes these as literals inside the expressions themselves, so they are
# named here only so tests/test_constants.py can pin them to that source.  Same doubles.
TROP_P0, TROP_DP = 30000.0, 21500.0  # tropopause_pressure = 30000.0 - 21500.0 cos^2(lat), Pa
OH_A, OH_EA = 2.45e-12, 1775.0  # krate_oh = 2.45e-12 exp(-1775.0/T)
CL_A, CL_T0, CL_N, CL_EA = 2.36e-12, 298.0, 1.37, 939.0  # krate_cl_trop


def _descending(levels, what):
    if not (np.diff(levels, axis=-1) < 0.0).all():
        raise ValueError(f"rebin: {what} levels not strictly descending (the Fortran keeps the stale column)")


def rebin(ip_levels, ip_data, op_levels):
    """rebin_levels(ip_mix=.true.): the layer-mean of `ip_data` (…, nip) given on the edges
    `ip_levels` (…, nip+1) over each layer of `op_levels` (…, nop+1), all descending.
    Leading axes broadcast."""
    return rebin_many(ip_levels, (ip_data,), op_levels)[0]


def rebin_many(ip_levels, ip_datas, op_levels):
    """`rebin` of several columns on the same edges, the overlaps worked out once."""
    _descending(ip_levels, "input")
    _descending(op_levels, "output")
    lead = np.broadcast_shapes(ip_levels.shape[:-1], *(d.shape[:-1] for d in ip_datas), op_levels.shape[:-1])
    ip_levels = np.broadcast_to(ip_levels, (*lead, ip_levels.shape[-1]))
    op_levels = np.broadcast_to(op_levels, (*lead, op_levels.shape[-1]))
    op_bot, op_top = op_levels[..., :-1], op_levels[..., 1:]  # (…, nop)
    nip = ip_levels.shape[-1] - 1
    # only input layers first..last-1 can overlap an output layer (both columns descend): the
    # ones below its floor or above its top add exactly zero, so they are skipped, in order
    first = (ip_levels[..., None, 1:] >= op_bot[..., None]).sum(axis=-1)  # (…, nop)
    last = (ip_levels[..., None, :-1] > op_top[..., None]).sum(axis=-1)
    terms = []
    for d in range(int((last - first).max(initial=0))):
        k = first + d
        kk = np.minimum(k, nip - 1)
        ip_bot = np.take_along_axis(ip_levels, kk, axis=-1)
        ip_top = np.take_along_axis(ip_levels, kk + 1, axis=-1)
        thickness = np.minimum(op_bot, ip_bot) - np.maximum(op_top, ip_top)
        terms.append((kk, (k < last) & (thickness > 0.0), thickness))
    width = op_bot - op_top
    out = []
    for ip_data in ip_datas:
        ip_data = np.broadcast_to(ip_data, (*lead, nip))
        overlap = np.zeros(op_bot.shape)
        for kk, inside, thickness in terms:
            overlap = overlap + np.where(inside, np.take_along_axis(ip_data, kk, axis=-1) * thickness, 0.0)
        out.append(np.divide(overlap, width, out=overlap, where=width > 0.0))
    return out


def strat_edges(press):
    """Edges (…, n+1) in Pa of the zonal loss-rate columns given at pressures `press` (…, n) hPa."""
    n = press.shape[-1]
    edges = np.empty((*press.shape[:-1], n + 1))
    edges[..., 0] = press[..., 0] + 0.5 * (press[..., 0] - press[..., 1])
    edges[..., 1:n] = (press[..., :-1] + press[..., 1:]) / 2.0
    edges[..., n] = 0.0
    return edges * 100.0


def tropopause(lat):
    """Pressure (Pa) of the tropopause at latitude `lat` (radians)."""
    c = libm.cos(lat)
    return TROP_P0 - TROP_DP * (c * c)


def _column(edges, data, phlb, first_value):
    """Rebin the column `data` (given on `edges`) onto the model layers `phlb` (im, jm, lm+1),
    then fill the layers whose floor lies below the first edge with `first_value`."""
    col = rebin(edges, data, phlb)
    return np.where(phlb[..., :-1] > edges[..., :1], first_value[..., None], col)


def loss_rates(
    temper, phlb, lat, strat_press, strat_oh, strat_cl, strat_od, oh_3d,
    pl_trop=None, cl_trop=None, soil=None, oh_scale=1.0, cl_scale=1.0, o1d_scale=1.0, soil_rate=0.0,
):  # fmt: skip
    """ch4_loss_kernel: the loss rates (im, jm, lm, 4) of the month.  `strat_*` are the zonal
    (jm, nstrat) columns, `pl_trop`/`cl_trop` (im, jm, ntrop) the tropospheric Cl (None: not
    applied), `soil` (im, jm) the soil sink map (None: not applied)."""
    im, jm, lm = temper.shape
    edges = strat_edges(strat_press)[None]  # (1, jm, nstrat+1): the same column for every i
    columns = {"oh": strat_oh, "cl": strat_cl, "od": strat_od}
    rebinned = rebin_many(edges, [data[None] for data in columns.values()], phlb)
    below = phlb[..., :-1] > edges[..., :1]
    strat = {name: np.where(below, data[None, :, 0][..., None], col)
             for (name, data), col in zip(columns.items(), rebinned)}  # fmt: skip

    cl_trop_col = np.zeros((im, jm, lm))
    if cl_trop is not None:
        trop_edges = np.concatenate([pl_trop, np.zeros((im, jm, 1))], axis=-1)
        cl_trop_col = _column(trop_edges, cl_trop, phlb, cl_trop[..., 0])
        krate = (CL_A * libm.pow(temper / CL_T0, CL_N)) * libm.exp(-CL_EA / temper)
        cl_trop_col = (cl_trop_col * krate) * cl_scale

    krate_oh = OH_A * libm.exp(-OH_EA / temper)
    oh_spiv = (krate_oh * oh_3d) * oh_scale

    stratospheric = phlb[..., :-1] < tropopause(lat)[None, :, None]
    loss = np.empty((im, jm, lm, 4))
    loss[..., 0] = np.where(stratospheric, strat["oh"], oh_spiv)
    loss[..., 1] = np.where(stratospheric, strat["cl"], strat["cl"] + cl_trop_col)
    loss[..., 2] = strat["od"] * o1d_scale
    loss[..., 3] = 0.0
    if soil is not None:
        loss[..., 0, 3] = -(soil_rate * soil)
    return loss


SETTINGS = ("oh_scale", "cl_scale", "o1d_scale", "trop_cl_sink", "soil_sink", "soil_rate")  # the rc's


def loss_inputs(temper, phlb, lat, fields, settings):
    """loss_rates' kwargs: `fields` the month's (oh_3d, strat_*, and pl_trop / cl_trop, soil as
    the sinks are on), `settings` a mapping of SETTINGS."""
    inputs = {
        "temper": temper, "phlb": phlb, "lat": lat, "oh_3d": fields["oh_3d"],
        **{f"strat_{k}": fields[f"strat_{k}"] for k in ("press", "oh", "cl", "od")},
        **{k: settings[k] for k in ("oh_scale", "cl_scale", "o1d_scale")},
    }  # fmt: skip
    if settings["trop_cl_sink"]:
        inputs.update(pl_trop=fields["pl_trop"], cl_trop=fields["cl_trop"])
    if settings["soil_sink"]:
        inputs.update(soil=fields["soil"], soil_rate=settings["soil_rate"])
    return inputs


def decay(loss, dt_sec):
    """Phase 1: the per-cell decay factor over dt_sec of the summed reactions."""
    return libm.exp(-(seqsum(loss, axis=-1) * dt_sec))


def apply_decay(state, factor, tracers):
    """Phase 2: scale the mass and slopes of the (1-based) `tracers` by `factor` (im, jm, lm)
    in place.  The Fortran skips cells with factor == 1.0; multiplying them is the same bits."""
    for t in tracers:
        for name in ("rm", "rxm", "rym", "rzm"):
            state[name][..., t - 1] *= factor
