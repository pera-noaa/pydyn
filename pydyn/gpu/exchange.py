"""zoom.exchange and zoom.advect's slicing on the device as box programs (boxes.py): the
exchange's copies and scalings (put_x/yedges), the compress / uncompress around a child's x and
y advection, the outer interface faces zeroed in the second half of a parent step, and the
periodic halo copies of region 1's rows.  Each function is its reference's statements, staged
where a statement reads what an earlier one wrote, built once per state and call site and then
run as a few dispatches.
"""

from contextlib import contextmanager

from . import boxes
from .boxes import ADV, COPY, DIV, DIVMUL, MUL, SEQSUM, SET, Batch, Raw

__all__ = [
    "compress_xedges",
    "compress_yedges",
    "mass_rows",
    "outer_x_faces_zeroed",
    "outer_y_faces_zeroed",
    "periodic_rows",
    "put_xedges",
    "put_yedges",
    "reduced_x_after",
    "reduced_x_before",
    "reduced_xm_after",
    "uncompress_xedges",
    "uncompress_yedges",
]

TRACERS = ("rm", "rxm", "rym", "rzm")


def _cache(st):
    """The state's programs: they hold its arrays' addresses, so they live with it."""
    c = getattr(st, "_pydyn_programs", None)
    if c is None:
        c = st._pydyn_programs = {}
    return c


def _run(st, key, build):
    boxes.program(_cache(st), key, build)


run = _run


def zero(st, key, view):
    """view[...] = 0 (a whole array): a box program."""
    _run(st, ("zero", key), lambda: ([_one(SET, view)], ()))


def _one(op, dst, *srcs, **kw):
    b = Batch()
    b.add(op, dst, *srcs, **kw)
    return b


def coarsen(st, region, build):
    _run(st, ("coarsen_region", region), build)


def save(st, key, views):
    """Copies of views into scratch kept with the program, for restore(st, key)."""

    def build():
        from .backend import backend

        xp, b, keep = backend().xp, Batch(), []
        for v in views:
            tmp = xp.zeros(v.shape, dtype=v.dtype)
            b.add(COPY, tmp, v)
            keep.append((v, tmp))
        return [b], keep

    _run(st, ("save", key), build)


def restore(st, key):
    def build():
        b = Batch()
        for v, tmp in _cache(st)[("save", key)].keep:
            b.add(COPY, v, tmp)
        return [b], ()

    _run(st, ("restore", key), build)


def scale(st, key, views, factor):
    """v[...] = v * factor for each view."""

    def build():
        b = Batch()
        for v in views:
            b.add(MUL, v, v, scalar=factor)
        return [b], ()

    _run(st, ("scale", key), build)


# ------------------------------------------------------------------ put_xedges / put_yedges
def put_xedges(zoom, st, region, tracers=True):
    _run(st, ("put_xedges", region, tracers), lambda: ([_put_xedges(zoom, st, region, tracers)], ()))


def _put_xedges(zoom, st, region, tracers):
    b, f = Batch(), st[region]
    for child in zoom.children[region]:
        gc = zoom.g(child)
        xr, yr, zr = zoom.ref(child, "x"), zoom.ref(child, "y"), zoom.ref(child, "z")
        if gc.im // gc.xref == zoom.g(1).im:
            continue
        if zr != 1:
            raise NotImplementedError("pydyn.gpu.put_xedges: zooming in z")
        c = st[child]
        jmc, lmc = gc.jm, zoom.lm
        yzref = 1.0 / (yr * zr)
        xyzref = 1.0 / (xr * yr * zr)
        jb, lb = zoom.jbeg[child], zoom.lbeg[child]
        rows, levs = slice(jb, jb + (jmc - 1) // yr), slice(lb, lb + lmc - 1)  # jp, lp
        ib, ie = zoom.ibeg[child], zoom.iend[child]
        plan = [("m", (), xyzref)]
        if tracers:
            plan += [
                (name, (n,), inner)
                for name, inner in zip(TRACERS, (xyzref, yzref, yzref, yzref))
                for n in range(1, st.nt + 1)
            ]
        for name, t, inner in plan:
            q, qc = f[name], c[name]
            walls = [(0, ib - 1, yzref), (gc.im + 1, ie + 1, yzref)]
            for i in range(1, xr + 1):
                walls += [(i, ib, inner), (gc.im + 1 - i, ie, inner)]
            for i_child, i_parent, scale in walls:
                dst = qc[(i_child, slice(1, jmc), slice(1, lmc), *t)]
                b.add(MUL, dst, q[(i_parent, rows, levs, *t)], scalar=scale, rep=(yr, 1))
    return b


def put_yedges(zoom, st, region, tracers=True):
    _run(st, ("put_yedges", region, tracers), lambda: ([_put_yedges(zoom, st, region, tracers)], ()))


def _put_yedges(zoom, st, region, tracers):
    b, f = Batch(), st[region]
    im_p = zoom.g(region).im
    for child in zoom.children[region]:
        gc = zoom.g(child)
        xr, yr, zr = zoom.ref(child, "x"), zoom.ref(child, "y"), zoom.ref(child, "z")
        if zr != 1:
            raise NotImplementedError("pydyn.gpu.put_yedges: zooming in z")
        c = st[child]
        imc, lmc = gc.im, zoom.lm
        xzref = 1.0 / (xr * zr)
        xyzref = 1.0 / (xr * yr * zr)
        ib, lb = zoom.ibeg[child], zoom.lbeg[child]
        if ib - 1 + (imc - 1) // xr >= im_p:
            raise NotImplementedError("pydyn.gpu.put_yedges: a child across the parent's x seam")
        cols, levs = slice(ib, ib + (imc - 1) // xr), slice(lb, lb + lmc - 1)  # ip, lp
        jb, je = zoom.jbeg[child], zoom.jend[child]
        plan = [("m", (), xyzref)]
        if tracers:
            plan += [
                (name, (n,), inner)
                for name, inner in zip(TRACERS, (xyzref, xzref, xzref, xzref))
                for n in range(1, st.nt + 1)
            ]
        for name, t, inner in plan:
            q, qc = f[name], c[name]
            walls = [(0, jb - 1, xzref), (gc.jm + 1, je + 1, xzref)]
            for j in range(1, yr + 1):
                walls += [(j, jb, inner), (gc.jm + 1 - j, je, inner)]
            for j_child, j_parent, scale in walls:
                dst = qc[(slice(1, imc), j_child, slice(1, lmc), *t)]
                b.add(MUL, dst, q[(cols, j_parent, levs, *t)], scalar=scale, rep=(xr, 1))
    return b


# ------------------------------------------------------------------ compress / uncompress
def _x_active(zoom, region):
    xr, g = zoom.ref(region, "x"), zoom.g(region)
    return not (xr == 1 or g.im // g.xref == zoom.g(1).im)


def compress_xedges(zoom, st, region, js, je, ls, le, tracers=True):
    if _x_active(zoom, region):
        _run(
            st,
            ("compress_x", region, js, je, ls, le, tracers),
            lambda: (_compress(zoom, st, region, 0, (js, je), (ls, le), tracers), ()),
        )


def compress_yedges(zoom, st, region, is_, ie, ls, le, tracers=True):
    if not (zoom.ref(region, "y") == 1 or region == 1):
        _run(
            st,
            ("compress_y", region, is_, ie, ls, le, tracers),
            lambda: (_compress(zoom, st, region, 1, (is_, ie), (ls, le), tracers), ()),
        )


def _compress(zoom, st, region, axis, other, levs, tracers):
    """_compress_x/yedges: the seqsums first (they read the cells the copies write), then the
    halo copies."""
    r = zoom.ref(region, "xy"[axis])
    n = (zoom.g(region).im, zoom.g(region).jm)[axis]
    f = st[region]
    o, lv = slice(*other), slice(*levs)

    def sec(q, k):
        return q[(k, o, lv)] if axis == 0 else q[(o, k, lv)]

    sums, copies = Batch(), Batch()
    for name in ("m", "rm") if tracers else ("m",):
        q = f[name]
        for dst, lo, hi in ((r, 1, r), (n - r + 1, n - r + 1, n)):
            sums.add(SEQSUM, sec(q, dst), sec(q, slice(lo, hi)), sum_axis=axis)
    for name in ("m", *TRACERS) if tracers else ("m",):
        q = f[name]
        copies.add(COPY, sec(q, r - 1), sec(q, 0))
        copies.add(COPY, sec(q, n - r + 2), sec(q, n + 1))
    return [sums, copies]


def uncompress_xedges(zoom, st, region, js, je, ls, le, tracers=True):
    if _x_active(zoom, region):
        _run(
            st,
            ("uncompress_x", region, js, je, ls, le, tracers),
            lambda: (_uncompress(zoom, st, region, 0, (js, je), (ls, le), tracers), ()),
        )


def uncompress_yedges(zoom, st, region, is_, ie, ls, le, tracers=True):
    if not (zoom.ref(region, "y") == 1 or region == 1):
        _run(
            st,
            ("uncompress_y", region, is_, ie, ls, le, tracers),
            lambda: (_uncompress(zoom, st, region, 1, (is_, ie), (ls, le), tracers), ()),
        )


def _uncompress(zoom, st, region, axis, other, levs, tracers):
    """q[lo:hi] = q[edge] / r: every row but the edge's from it first, then the edge's own."""
    r = zoom.ref(region, "xy"[axis])
    n = (zoom.g(region).im, zoom.g(region).jm)[axis]
    f = st[region]
    o, lv = slice(*other), slice(*levs)

    def sec(q, k):
        return q[(k, o, lv)] if axis == 0 else q[(o, k, lv)]

    rows, edges = Batch(), Batch()
    for lo, hi, edge in ((1, r, r), (n - r + 1, n, n - r + 1)):
        for name in ("m", *TRACERS) if tracers else ("m",):
            q = f[name]
            for k in range(lo, hi + 1):
                if k != edge:
                    rows.add(DIV, sec(q, k), sec(q, edge), scalar=float(r))
            edges.add(DIV, sec(q, edge), sec(q, edge), scalar=float(r))
    return [rows, edges]


# ------------------------------------------------------------------ outer interface faces
@contextmanager
def outer_x_faces_zeroed(zoom, st, region, second):
    g, xr = zoom.g(region), zoom.ref(region, "x")
    zero = second and g.im // g.xref < zoom.g(1).im
    if zero:
        _run(
            st,
            ("outer_x", region, "zero"),
            lambda: _faces_zero(st, region, "am", [(xr - 1, 0), (g.im - xr + 1, 0)]),
        )
    yield
    if zero:
        _run(st, ("outer_x", region, "restore"), lambda: _faces_restore(st, ("outer_x", region, "zero")))


@contextmanager
def outer_y_faces_zeroed(zoom, st, region, second):
    g, yr = zoom.g(region), zoom.ref(region, "y")
    faces = []
    if second:
        if g.touch_sp != 1:
            faces.append((yr, 1))
        if g.touch_np != 1:
            faces.append((g.jm - yr + 2, 1))
    if faces:
        _run(st, ("outer_y", region, "zero"), lambda: _faces_zero(st, region, "bm", faces))
    yield
    if faces:
        _run(st, ("outer_y", region, "restore"), lambda: _faces_restore(st, ("outer_y", region, "zero")))


def _face(q, k, axis):
    return q[k, :, :] if axis == 0 else q[:, k, :]


def _faces_zero(st, region, name, faces):
    """Save each face into scratch, then zero it; the scratch is kept for the restore."""
    from .backend import backend

    xp, q = backend().xp, st[region][name]
    save, zero, keep = Batch(), Batch(), []
    for k, axis in faces:
        v = _face(q, k, axis)
        tmp = xp.zeros(v.shape, dtype=v.dtype)
        save.add(COPY, tmp, v)
        zero.add(SET, v, scalar=0.0)
        keep.append((v, tmp))
    return [save, zero], keep


def _faces_restore(st, zero_key):
    back = Batch()
    for v, tmp in _cache(st)[zero_key].keep:
        back.add(COPY, v, tmp)
    return [back], ()


# ------------------------------------------------------------------ region 1's periodic rows
def periodic_rows(zoom, st, runs):
    """_dynamu_region1's halo copies: for each run (j0, j1, iimr) and every field,
    q[0, j0:j1] = q[iimr, j0:j1] and q[iimr+1, j0:j1] = q[1, j0:j1] (no copy reads another's
    destination: one stage)."""
    _run(st, ("periodic_rows", tuple(runs)), lambda: ([_periodic(st, runs)], ()))


def _periodic(st, runs):
    b, f = Batch(), st[1]
    for j0, j1, iimr in runs:
        for name in (*TRACERS, "m"):
            q = f[name]
            b.add(COPY, q[0, j0:j1], q[iimr, j0:j1])
            b.add(COPY, q[iimr + 1, j0:j1], q[1, j0:j1])
    return b


# ------------------------------------------------------------------ region 1's reduced grid
def reduced_x_before(zoom, st, fields, am, tracers=True):
    """_dynamu_region1's m_uni, am_uni = copies; uni2red (uni2red_mass when not tracers);
    uni2red_mf.  The copies and the cluster sums read the uniform rows into scratch first,
    then the reduced rows are written, then am's periodic face."""
    _run(st, ("reduced_x_before", tracers), lambda: _reduced_before(zoom, fields, am, tracers))


def _scratch_like(v):
    from .backend import backend

    return backend().xp.zeros(v.shape, dtype=v.dtype)


def _reduced_before(zoom, fields, am, tracers):
    from .backend import backend

    be = backend()
    grid = zoom.g(1).redgrid
    m = fields[0]
    m_uni, am_uni = _scratch_like(m), _scratch_like(am)
    read, write, face, keep = Batch(), Batch(), Batch(), {"m_uni": m_uni, "am_uni": am_uni, "tmp": []}
    read.add(COPY, m_uni, m)
    read.add(COPY, am_uni, am)
    qs = fields if tracers else fields[:1]
    for j, c, n in grid.bands:
        for k, q in enumerate(qs):
            row = q[:, j]
            st = be.strides(row)
            tmp = be.xp.zeros((n, *row.shape[1:]), dtype=row.dtype)
            keep["tmp"].append(tmp)
            clusters = Raw(be.address(row), (n, c, *row.shape[1:]), (c * st[0], st[0], *st[1:]))
            read.add(SEQSUM, tmp, clusters, sum_axis=1)
            write.add(COPY, row[:n], tmp)
            if k > 0:  # uni2red zeroes the tracers' unused cells, not m's
                write.add(SET, row[n:], scalar=0.0)
        arow = am[:, j]
        ast = be.strides(arow)
        tmp = be.xp.zeros((n, *arow.shape[1:]), dtype=arow.dtype)
        keep["tmp"].append(tmp)
        read.add(
            COPY,
            tmp,
            Raw(be.address(arow) + c * ast[0] * arow.itemsize, (n, *arow.shape[1:]), (c * ast[0], *ast[1:])),
        )  # row[c * arange(1, n + 1)]
        write.add(COPY, arow[1 : n + 1], tmp)
        write.add(SET, arow[n + 1 :], scalar=-1.0)
        face.add(COPY, arow[0], arow[n])
    return [read, write, face], keep


def reduced_x_after(zoom, st, fields, am):
    """red2uni(grid, *fields, advance(m_uni, am_uni)); am[...] = am_uni; the reduced rows'
    periodic halos q[0, j0 + 1] = q[im, j0 + 1]."""
    before = _cache(st)[("reduced_x_before", True)].keep
    _run(st, ("reduced_x_after",), lambda: _reduced_after(zoom, st, fields, am, before))


def _advanced(be, m, before):
    """m_adv = advance(m_uni, am_uni): (m_uni + am_uni[:-1]) - am_uni[1:]."""
    m_adv = _scratch_like(m)
    b = Batch()
    b.add(ADV, m_adv, before["m_uni"], before["am_uni"][:-1], before["am_uni"][1:])
    return m_adv, b


def _reduced_after(zoom, st, fields, am, before):
    from .backend import backend

    be = backend()
    g, grid = zoom.g(1), zoom.g(1).redgrid
    m = fields[0]
    m_adv, adv = _advanced(be, m, before)
    keep = {"m_adv": m_adv, "tmp": []}
    spread, mass, halo = Batch(), Batch(), Batch()
    for j, c, n in grid.bands:
        for q in fields[1:]:
            tmp = be.xp.zeros((n, *q.shape[2:]), dtype=q.dtype)
            keep["tmp"].append(tmp)
            adv.add(COPY, tmp, q[:n, j])  # q(:n) read before it is overwritten
            spread.add(
                DIVMUL,
                q[: n * c, j],
                m_adv[: n * c, j][..., None],
                m[:n, j][..., None],
                tmp,
                rep=[None, (c, 1, 1), (c, 1, 1)],
            )
        mass.add(COPY, m[:, j], m_adv[:, j])  # red2uni_mass, after the scale has read m(:n)
    spread.add(COPY, am, before["am_uni"])
    f = st[1]
    for j0 in grid.rows:
        for name in (*TRACERS, "m"):
            q = f[name]
            halo.add(COPY, q[0, j0 + 1], q[g.im, j0 + 1])
    return [adv, spread, mass, halo], keep


def reduced_xm_after(zoom, st, m, am):
    """_dynamum_region1_mass: red2uni_mass(grid, m, advance(m_uni, am_uni)); am[...] = am_uni."""
    before = _cache(st)[("reduced_x_before", False)].keep
    _run(st, ("reduced_xm_after",), lambda: _reduced_m_after(zoom, m, am, before))


def _reduced_m_after(zoom, m, am, before):
    from .backend import backend

    m_adv, adv = _advanced(backend(), m, before)
    mass = Batch()
    for j, _, _ in zoom.g(1).redgrid.bands:
        mass.add(COPY, m[:, j], m_adv[:, j])
    mass.add(COPY, am, before["am_uni"])
    return [adv, mass], {"m_adv": m_adv}


def mass_rows(zoom, st, js, je):
    """_dynamum_region1_mass' halos: m[0] = m[im] on every row, m[im+1] = m[1] on the rows
    the reduced grid leaves whole."""
    _run(st, ("mass_rows", js, je), lambda: ([_mass_rows(zoom, st, js, je)], ()))


def _mass_rows(zoom, st, js, je):
    g, m, b = zoom.g(1), st[1]["m"], Batch()
    reduced = {j + 1 for j in g.redgrid.rows} if g.redgrid is not None else set()
    for j in range(js, je + 1):
        b.add(COPY, m[0, j], m[g.im, j])
        if j not in reduced:
            b.add(COPY, m[g.im + 1, j], m[1, j])
    return b
