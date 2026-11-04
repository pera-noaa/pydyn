"""pydyn.kernels on device arrays: the same calls, the same in-place semantics, the work done by
csrc/ kernels.  Each function takes the views the reference function takes (any strides) and
leaves them as the reference leaves them; the host sees only the sub-cycle counts and error
flags ("control" transfers of a few bytes).

The sub-cycle caps are read off the reference functions' signatures, so there is one copy of
each and tests/test_constants.py pins it.
"""

import inspect

import numpy as np

from .. import kernels as K
from ..kernels import CFLError
from .backend import backend

__all__ = [
    "advect_x_rows",
    "advect_y",
    "advect_z",
    "advect_z_substeps",
    "cluster_sum",
    "convec",
    "fluxes_from_meteo",
]


def _default(f, name):
    return inspect.signature(f).parameters[name].default


def _flag(dev):
    """A device int flag (or small int array) read back: a control transfer."""
    return backend().to_host(dev, "control")


def _zeros(shape, dtype=None):
    be = backend()
    return be.xp.zeros(shape, dtype=be.real if dtype is None else dtype)


def _scratch(shape, dtype=None, fill=0):
    """An interval's scratch (Backend.scratch): the same address every speculated interval."""
    return backend().scratch(shape, dtype, fill)


def _tracer_strides(rm, rxm, rym, rzm):
    st = [backend().strides(q, 4) for q in (rm, rxm, rym, rzm)]
    if any(s != st[0] for s in st[1:]):
        raise ValueError("pydyn.gpu: rm, rxm, rym, rzm views must share their strides")
    return st[0]


NT_MAX = 16  # csrc/pd.h PD_NT_MAX: the tracers a thread holds face values for


def _check_nt(nt):
    if nt > NT_MAX:
        raise ValueError(f"pydyn.gpu: {nt} tracers, the kernels hold {NT_MAX} (PD_NT_MAX)")


# ------------------------------------------------------------------ x
def advect_x_rows(m, rm, rxm, rym, rzm, a, nrow, limits=True, max_nloop=None, deferred=False):
    """kernels._advect_rows_x of every (j, l) row of a region at once: m (n, jm, lm), tracers
    (n, jm, lm, nt), a (n+1, jm, lm) with a[0] == a[n] per row, nrow (jm,) int32 the cells of
    each latitude (imred[j] on a reduced row); deferred: its CFLError raised at the next sync
    (Backend.check).  Returns the device arrays (nloop, alpha_max),
    one value per row j * lm + l."""
    be = backend()
    max_nloop = _default(K.advect_x, "max_nloop") if max_nloop is None else max_nloop
    _, jm, lm = m.shape
    nt = rm.shape[3]
    _check_nt(nt)
    nloop, amax, fail = _scratch(jm * lm, np.int32), _scratch(jm * lm), _scratch(1, np.int32)
    be.launch(
        "pd_x_rows",
        jm * lm,
        jm,
        lm,
        nt,
        nrow,
        m,
        *be.strides(m),
        rm,
        rxm,
        rym,
        rzm,
        *_tracer_strides(rm, rxm, rym, rzm),
        a,
        *be.strides(a),
        int(limits),
        max_nloop - 1,
        nloop,
        amax,
        fail,
    )
    be.check(fail, CFLError("dynamu: nloop too high"), deferred)
    return nloop, amax


# ------------------------------------------------------------------ y
def advect_y(m, rm, rxm, rym, rzm, bm, limits=True, max_nloop=None, stats=None, full_bm=None, level_offset=0):
    """kernels.advect_y: m (im, jm, lm), bm (im, jm+1, lm).  The search runs per layer on the
    device (pd_y_search to its fixpoint); `full_bm` (default bm itself) is the array whose whole
    layers dynamv scales and restores -- region 1's halo'd bm, so that zoom._rescale_layer_halo
    is part of the same two passes -- with its levels offset by `level_offset` from bm's.
    Returns the device array of the layers' nloop."""
    be = backend()
    max_nloop = _default(K.advect_y, "max_nloop") if max_nloop is None else max_nloop
    im, jm, lm = m.shape
    nt = rm.shape[3]
    _check_nt(nt)
    nloop, _ = _search(
        lambda cand, nxt, bad: be.launch(
            "pd_y_search",
            im * lm,
            im,
            jm,
            lm,
            m,
            *be.strides(m),
            bm,
            *be.strides(bm),
            cand,
            max_nloop - 1,
            nxt,
            bad,
        ),
        lm,
        max_nloop - 1,
        "dynamv: nloop too high",
        "advect_y: non-zero flux through the polar outer faces",
    )
    full = bm if full_bm is None else full_bm
    n0, n1 = full.shape[0], full.shape[1]
    be.launch("pd_layer_scale", n0 * n1 * lm, n0, n1, level_offset, lm, full, *be.strides(full), nloop, 0)
    bmax = _scratch(lm)
    be.launch(
        "pd_y_steps",
        im * lm,
        im,
        jm,
        lm,
        nt,
        m,
        *be.strides(m),
        rm,
        rxm,
        rym,
        rzm,
        *_tracer_strides(rm, rxm, rym, rzm),
        bm,
        *be.strides(bm),
        nloop,
        int(limits),
        bmax,
    )
    be.launch("pd_layer_scale", n0 * n1 * lm, n0, n1, level_offset, lm, full, *be.strides(full), nloop, 1)
    if stats is not None:
        nl, bx = be.to_host(nloop, "control"), be.to_host(bmax, "control")
        stats["nloop_max"], stats["xi"] = max(1, int(nl.max())), float(bx.max())
    return nloop


SEARCH_ROUNDS = 2  # rounds queued a read: a round from the fixpoint returns it, and reads only


def _search(round_, n, nmax, cfl_message, bad_message=None):
    """The fixpoint of a sub-cycle search: cand = 1, then each round's per-line maxima, until a
    round returns cand itself.  `n` candidates (one per layer, or one for the region).
    Returns (device, host) copies of the result.  SEARCH_ROUNDS rounds are queued before the
    host reads them (one sync for all): a round started from the fixpoint returns it, and the
    search kernels write nothing but their candidates, so the rounds past it change nothing."""
    be = backend()
    had = np.ones(n, dtype=np.int32)
    cand = _scratch(n, np.int32, fill=1)
    while True:
        rounds = []
        for _ in range(SEARCH_ROUNDS):
            nxt, bad = _scratch(n, np.int32), _scratch(1, np.int32)
            round_(cand, nxt, bad)
            rounds.append((cand, nxt, bad))
            cand = nxt
        if getattr(be, "speculating", False):  # the last round is the fixpoint: checked later
            be.assume(lambda rounds=rounds: _converged(rounds, nmax, bad_message is not None))
            return rounds[-1][1], None
        for c, nxt, bad in rounds:
            if bad_message is not None and _flag(bad)[0]:
                raise ValueError(bad_message)
            got = be.to_host(nxt, "control")
            if (got > nmax).any():
                raise CFLError(cfl_message)
            if (got == had).all():
                return c, got
            had = got


def _converged(rounds, nmax, has_bad):
    """_search's verdict on the rounds queued, read after the sync: the fixpoint reached, under
    nmax, no bad flag -- the speculation's assumption."""
    be = backend()
    had = np.ones(be.to_host(rounds[0][1], "control").shape, dtype=np.int32)
    for _, nxt, bad in rounds:
        if has_bad and int(be.to_host(bad, "control")[0]):
            return False
        got = be.to_host(nxt, "control")
        if (got > nmax).any():
            return False
        if (got == had).all():
            return True
        had = got
    return False


def ok_flag(fail):
    """Check_CFL's verdict on a device fail flag: read now, or assumed fine while speculating
    (zoom.speculate) and checked at its end."""
    be = backend()
    if getattr(be, "speculating", False):
        be.assume(lambda: not int(be.to_host(fail, "control")[0]))
        return True
    return not int(be.to_host(fail, "control")[0])


# ------------------------------------------------------------------ z
def advect_z(m, rm, rxm, rym, rzm, cm, limits=True, iter_limit=None, stats=None):
    """kernels.advect_z: m (im, jm, lm), cm (im, jm, lm+1); one n_advz for the region, found on
    the device.  Returns n_advz.  stats gets nloop_max; its 'xi' (the largest Courant number
    the Fortran's search saw, up to its first violation) is not computed here: None."""
    be = backend()
    iter_limit = _default(K.advect_z, "iter_limit") if iter_limit is None else iter_limit
    im, jm, lm = m.shape
    dev_n, _ = _search(
        lambda cand, nxt, bad: be.launch(
            "pd_z_search",
            im * jm,
            im,
            jm,
            lm,
            m,
            *be.strides(m),
            cm,
            *be.strides(cm),
            cand,
            iter_limit,
            nxt,
            bad,
        ),
        1,
        iter_limit,
        "dynamw: too many CFL violations!",
        "dynamw: non-zero halo interfaces in cm",
    )
    advect_z_substeps(m, rm, rxm, rym, rzm, cm, dev_n, limits, prelimit=limits)
    if stats is not None:
        n = int(be.to_host(dev_n, "control")[0])
        stats["nloop_max"], stats["xi"] = n, None
    return dev_n


def advect_z_substeps(m, rm, rxm, rym, rzm, cm, n_advz, limits=True, prelimit=False):
    """kernels.advect_z_substeps (cm is read as cm / n_advz and never written); prelimit applies
    dynamw's limiter first, which advect_z does before its search.  n_advz a host int or the
    search's device array."""
    be = backend()
    n_advz = _device_n(n_advz)
    im, jm, lm = m.shape
    nt = rm.shape[3]
    _check_nt(nt)
    be.launch(
        "pd_z_steps",
        im * jm,
        im,
        jm,
        lm,
        nt,
        m,
        *be.strides(m),
        rm,
        rxm,
        rym,
        rzm,
        *_tracer_strides(rm, rxm, rym, rzm),
        cm,
        *be.strides(cm),
        n_advz,
        int(limits),
        int(prelimit),
    )


def _device_n(n):
    """A sub-step count as pd_z_steps reads it: a device int32 array of one."""
    be = backend()
    return n if be.is_device(n) else be.to_device(np.array([int(n)], dtype=np.int32), "control")


# ------------------------------------------------------------------ dynam0
def fluxes_from_meteo(pu, pv, bt, dt, am, bm, cm):
    """kernels.fluxes_from_meteo into the given device views: am (im+1, jm, lm) = dt pu,
    bm (im, jm+1, lm) = dt pv, cm (im, jm, lm+1) levels 1..lm-1 from the convergence (levels 0
    and lm are not written).  bt a device (lm+1,) array."""
    be = backend()
    im, jm, lm = pu.shape[0] - 1, pu.shape[1], pu.shape[2]
    for dst, src in ((am, pu), (bm, pv)):
        n0, n1, n2 = src.shape
        be.launch("pd_scale_copy", n0 * n1 * n2, n0, n1, n2, dt, dst, *be.strides(dst), src, *be.strides(src))
    be.launch(
        "pd_dynam0_cm",
        im * jm,
        im,
        jm,
        lm,
        dt,
        pu,
        *be.strides(pu),
        pv,
        *be.strides(pv),
        bt,
        cm,
        *be.strides(cm),
    )


# ------------------------------------------------------------------ the reduced grid
def cluster_sum(q, n, c):
    """seqsum(clusters(q, n, c), axis=1) of a (cells, a[, b]) view: a new (n, a[, b]) array."""
    be = backend()
    v = q if q.ndim == 3 else q[..., None]
    out = _zeros((n, v.shape[1], v.shape[2]))
    be.launch(
        "pd_cluster_sum",
        n * v.shape[1] * v.shape[2],
        n,
        c,
        v.shape[1],
        v.shape[2],
        v,
        *be.strides(v),
        out,
        *be.strides(out),
    )
    return out if q.ndim == 3 else out[..., 0]


# ------------------------------------------------------------------ convec
INVERSE_KIND = {"crout": 0, "lapack": 1}


def convec(m, rm, rxm, rym, rzm, conv, dt, lmax, cols, inverse="crout"):
    """kernels.convec on the columns `cols` (device int32, flat i * jm + j) of the (im, jm, ...)
    views, in place.  conv: device entu detu entd detd (im, jm, lm), dkg (im, jm, >= lmax),
    cloud_top cloud_lfs (im, jm) int32.  inverse 'crout' is fastminv; 'lapack' the reference
    DGETRF + DGETRI (csrc/convec.cuh), the twin of the host library's dgetrf_ / dgetri_."""
    be = backend()
    if inverse not in INVERSE_KIND:
        raise ValueError(f"pydyn.gpu.convec: inverse {inverse!r}: 'crout' or 'lapack' (an open one)")
    jm = m.shape[1]
    nt = rm.shape[3]
    e = [conv[k] for k in ("entu", "detu", "entd", "detd")]
    st = be.strides(e[0])
    if any(be.strides(x) != st for x in e[1:]):
        raise ValueError("pydyn.gpu.convec: entu, detu, entd, detd must share their strides")
    ct, cl = conv["cloud_top"], conv["cloud_lfs"]
    info = _scratch(1, np.int32)
    lmaxc = max(8, -(-int(lmax) // 8) * 8)
    be.launch(
        "pd_convec",
        cols.size,
        cols.size,
        cols,
        jm,
        nt,
        int(lmax),
        dt,
        INVERSE_KIND[inverse],
        m,
        *be.strides(m),
        rm,
        rxm,
        rym,
        rzm,
        *_tracer_strides(rm, rxm, rym, rzm),
        *e,
        *st,
        conv["dkg"],
        *be.strides(conv["dkg"]),
        ct,
        cl,
        *be.strides(ct),
        info,
        defines={"PD_LMAXC": lmaxc},
    )
    if inverse == "lapack":  # only DGETRF / DGETRI set info
        be.check(
            info,
            ValueError("pydyn.gpu.convec: a singular convection matrix (dgetrf / dgetri info != 0)"),
            True,
        )


def inverse_rows(A):
    """The LAPACK kind's inverse (csrc/convec.cuh pd_inverse_rows) of each row-major n x n
    matrix of the device array A (ncol, n, n): (Y, info flag), for its tests."""
    be = backend()
    ncol, n, _ = A.shape
    Y, info = _zeros(A.shape), _zeros(1, np.int32)
    lmaxc = max(8, -(-int(n) // 8) * 8)
    be.launch("pd_inverse_rows", ncol, ncol, int(n), A, Y, info, defines={"PD_LMAXC": lmaxc})
    return Y, info


# ------------------------------------------------------------------ Check_CFL's mass-only kernels
def _no_tracers(m):
    """m itself stands in for the four tracer arrays (nt = 0: never read)."""
    return (m, m, m, m, 0, 0, 0, 0)


def mass_x_rows(m, a, nrow, max_nloop):
    """cfl.dynamum's rows: subcycle_rows and the mass-only sub-steps of every (j, l) row, the
    faces scaled and restored.  False when a row needs more than max_nloop - 1."""
    be = backend()
    _, jm, lm = m.shape
    nloop, amax, fail = _scratch(jm * lm, np.int32), _scratch(jm * lm), _scratch(1, np.int32)
    be.launch(
        "pd_x_rows",
        jm * lm,
        jm,
        lm,
        0,
        nrow,
        m,
        *be.strides(m),
        *_no_tracers(m),
        a,
        *be.strides(a),
        0,
        max_nloop - 1,
        nloop,
        amax,
        fail,
    )
    return ok_flag(fail)


def mass_y(m, bm, max_nloop):
    """cfl.dynamvm: one nloop per layer up to max_nloop, the mass-only sub-steps, bm scaled and
    multiplied back.  False when a layer needs more."""
    be = backend()
    im, jm, lm = m.shape
    try:
        nloop, _ = _search(
            lambda cand, nxt, bad: be.launch(
                "pd_y_search",
                im * lm,
                im,
                jm,
                lm,
                m,
                *be.strides(m),
                bm,
                *be.strides(bm),
                cand,
                max_nloop,
                nxt,
                bad,
            ),
            lm,
            max_nloop,
            "dynamvm",
        )
    except CFLError:
        return False
    n0, n1 = bm.shape[0], bm.shape[1]
    be.launch("pd_layer_scale", n0 * n1 * lm, n0, n1, 0, lm, bm, *be.strides(bm), nloop, 0)
    be.launch(
        "pd_y_steps",
        im * lm,
        im,
        jm,
        lm,
        0,
        m,
        *be.strides(m),
        *_no_tracers(m),
        bm,
        *be.strides(bm),
        nloop,
        0,
        _scratch(lm),
    )
    be.launch("pd_layer_scale", n0 * n1 * lm, n0, n1, 0, lm, bm, *be.strides(bm), nloop, 1)
    return True


def mass_z(m, cm, iter_limit):
    """cfl.dynamwm / zoom's column_trial on a whole region: one n_advz up to iter_limit, the
    mass advanced with cm / n_advz (cm not written).  False when more are needed."""
    be = backend()
    im, jm, lm = m.shape
    try:
        dev_n, _ = _search(
            lambda cand, nxt, bad: be.launch(
                "pd_z_search",
                im * jm,
                im,
                jm,
                lm,
                m,
                *be.strides(m),
                cm,
                *be.strides(cm),
                cand,
                iter_limit,
                nxt,
                bad,
            ),
            1,
            iter_limit,
            "dynamwm",
        )
    except CFLError:
        return False
    be.launch(
        "pd_z_steps",
        im * jm,
        im,
        jm,
        lm,
        0,
        m,
        *be.strides(m),
        *_no_tracers(m),
        cm,
        *be.strides(cm),
        dev_n,
        0,
        0,
    )
    return True


# ------------------------------------------------------------------ chemistry
def loss_rates(
    dt_sec,
    temper,
    phlb,
    lat,
    strat_press,
    strat_oh,
    strat_cl,
    strat_od,
    oh_3d,
    pl_trop=None,
    cl_trop=None,
    soil=None,
    oh_scale=1.0,
    cl_scale=1.0,
    o1d_scale=1.0,
    soil_rate=0.0,
):
    """chem.loss_rates and chem.decay on the device: (loss (im, jm, lm, 4), decay (im, jm, lm)).
    Host inputs are uploaded ("input"), device ones used as they are (phlb is the run's own;
    lat and strat_press stay host arrays: the tropopause and edges are derived from them).  The
    tropopause and the zonal columns' edges are computed on the host by chem's own functions
    (cos through ops.libm), the rates in csrc/pmath.cuh."""
    from .. import chem

    be = backend()

    def dev(x):
        if x is None or be.is_device(x):
            return x
        return be.to_device(np.asarray(x, dtype=float), "input")

    im, jm, lm = temper.shape
    nstrat = strat_press.shape[-1]
    k = be.to_device(
        np.array(
            [
                chem.OH_A,
                chem.OH_EA,
                chem.CL_A,
                chem.CL_T0,
                chem.CL_N,
                chem.CL_EA,
                oh_scale,
                cl_scale,
                o1d_scale,
                soil_rate,
                dt_sec,
            ],
            dtype=float,
        ),
        "input",
    )
    t, p, oh = dev(temper), dev(phlb), dev(oh_3d)
    pl, cl = dev(pl_trop), dev(cl_trop)
    trop = dev(chem.tropopause(np.asarray(lat, dtype=float)))
    edges = dev(chem.strat_edges(np.asarray(strat_press, dtype=float)))

    def dev_c(x):  # a contiguous copy on the device
        if x is None or be.is_device(x):
            return x if x is None else be.xp.ascontiguousarray(x)
        return dev(np.ascontiguousarray(x, dtype=float))

    s_oh, s_cl, s_od = (dev_c(x) for x in (strat_oh, strat_cl, strat_od))
    sl = dev_c(soil)
    loss, decay, bad = _zeros((im, jm, lm, 4)), _zeros((im, jm, lm)), _zeros(1, np.int32)
    cst = be.strides(cl) if cl is not None else (0, 0, 0)
    if pl is not None and be.strides(pl) != cst:
        raise ValueError("pydyn.gpu.loss_rates: pl_trop and cl_trop views differ in strides")
    ntrop = cl.shape[2] if cl is not None else 0
    be.launch(
        "pd_loss_rates",
        im * jm,
        im,
        jm,
        lm,
        k,
        t,
        *be.strides(t),
        p,
        *be.strides(p),
        trop,
        edges,
        nstrat,
        s_oh,
        s_cl,
        s_od,
        oh,
        *be.strides(oh),
        pl,
        cl,
        *cst,
        ntrop,
        sl,
        loss,
        decay,
        bad,
    )
    be.check(bad, ValueError("rebin: levels not strictly descending (the Fortran keeps the stale column)"))
    return loss, decay


def _check_lmk(lm):
    """The PD_LMK a column of lm levels compiles with (a multiple of 16)."""
    return max(16, -(-int(lm + 1) // 16) * 16)
