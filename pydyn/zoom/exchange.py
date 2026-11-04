"""The parent <-> child exchange: put/compress/uncompress_{x,y}edges, mix_edges, update_parent.
Part of pydyn.zoom (see its docstring)."""

import numpy as np

from ..ops import seqsum


# ------------------------------------------------------------------ put_xedges / put_yedges
def put_xedges(zoom, st, region, tracers=True):
    """advectx__slopes.F90 put_xedges: the parent's cells next to and on each x-wall of every
    child into the child's halo (0, im+1) and interface cells (1..xref_, im+1-xref_..im).
    tracers=False is advectm_cfl.F90's putm_xedges: the same statements for m alone."""
    f = st[region]
    for child in zoom.children[region]:
        gc = zoom.g(child)
        xr, yr, zr = zoom.ref(child, "x"), zoom.ref(child, "y"), zoom.ref(child, "z")
        if gc.im // gc.xref == zoom.g(1).im:
            continue  # [0, 360] wide: periodic boundaries instead
        c = st[child]
        jmc, lmc = gc.jm, zoom.lm
        yzref = 1.0 / (yr * zr)
        xyzref = 1.0 / (xr * yr * zr)
        jp = (zoom.jbeg[child] + np.arange(jmc) // yr)[:, None]
        lp = (zoom.lbeg[child] + np.arange(lmc) // zr)[None, :]
        ib, ie = zoom.ibeg[child], zoom.iend[child]

        m = f["m"]
        toc0 = m.at(ib - 1, jp, lp) * yzref
        toc1 = m.at(ib, jp, lp) * xyzref
        tocm = m.at(ie, jp, lp) * xyzref
        tocm1 = m.at(ie + 1, jp, lp) * yzref
        mc = c["m"]
        mc[0, 1:jmc, 1:lmc] = toc0
        mc[gc.im + 1, 1:jmc, 1:lmc] = tocm1
        for i in range(1, xr + 1):
            mc[i, 1:jmc, 1:lmc] = toc1
            mc[gc.im + 1 - i, 1:jmc, 1:lmc] = tocm

        if not tracers:
            continue
        for name, inner in (("rm", xyzref), ("rxm", yzref), ("rym", yzref), ("rzm", yzref)):
            q, qc = f[name], c[name]
            for n in range(1, st.nt + 1):
                toc0 = q.at(ib - 1, jp, lp, n) * yzref
                toc1 = q.at(ib, jp, lp, n) * inner  # slopes: "note the yzref..."
                tocm = q.at(ie, jp, lp, n) * inner
                tocm1 = q.at(ie + 1, jp, lp, n) * yzref
                qc[0, 1:jmc, 1:lmc, n] = toc0
                qc[gc.im + 1, 1:jmc, 1:lmc, n] = tocm1
                for i in range(1, xr + 1):
                    qc[i, 1:jmc, 1:lmc, n] = toc1
                    qc[gc.im + 1 - i, 1:jmc, 1:lmc, n] = tocm


def put_yedges(zoom, st, region, tracers=True):
    """advecty__slopes.F90 put_yedges: the same for the y-walls (children touching no pole);
    tracers=False is putm_yedges."""
    f = st[region]
    im_p = zoom.g(region).im
    for child in zoom.children[region]:
        gc = zoom.g(child)
        xr, yr, zr = zoom.ref(child, "x"), zoom.ref(child, "y"), zoom.ref(child, "z")
        c = st[child]
        imc, lmc = gc.im, zoom.lm
        xzref = 1.0 / (xr * zr)
        xyzref = 1.0 / (xr * yr * zr)
        ip = (np.mod(zoom.ibeg[child] - 1 + np.arange(imc) // xr, im_p) + 1)[:, None]
        lp = (zoom.lbeg[child] + np.arange(lmc) // zr)[None, :]
        jb, je = zoom.jbeg[child], zoom.jend[child]

        m = f["m"]
        toc0 = m.at(ip, jb - 1, lp) * xzref
        toc1 = m.at(ip, jb, lp) * xyzref
        tocm = m.at(ip, je, lp) * xyzref
        tocm1 = m.at(ip, je + 1, lp) * xzref
        mc = c["m"]
        mc[1:imc, 0, 1:lmc] = toc0
        mc[1:imc, gc.jm + 1, 1:lmc] = tocm1
        for j in range(1, yr + 1):
            mc[1:imc, j, 1:lmc] = toc1
            mc[1:imc, gc.jm + 1 - j, 1:lmc] = tocm

        if not tracers:
            continue
        for name, inner in (("rm", xyzref), ("rxm", xzref), ("rym", xzref), ("rzm", xzref)):
            q, qc = f[name], c[name]
            for n in range(1, st.nt + 1):
                toc0 = q.at(ip, jb - 1, lp, n) * xzref
                toc1 = q.at(ip, jb, lp, n) * inner  # slopes: "note the xzref..."
                tocm = q.at(ip, je, lp, n) * inner
                tocm1 = q.at(ip, je + 1, lp, n) * xzref
                qc[1:imc, 0, 1:lmc, n] = toc0
                qc[1:imc, gc.jm + 1, 1:lmc, n] = tocm1
                for j in range(1, yr + 1):
                    qc[1:imc, j, 1:lmc, n] = toc1
                    qc[1:imc, gc.jm + 1 - j, 1:lmc, n] = tocm


# ------------------------------------------------------------------ compress / uncompress
def _compress_xedges(zoom, st, region, js, je, ls, le, tracers=True):
    xr, g = zoom.ref(region, "x"), zoom.g(region)
    if xr == 1 or g.im // g.xref == zoom.g(1).im:
        return
    f, imr = st[region], g.im
    m = f["m"]
    m[xr, js:je, ls:le] = seqsum(m[1:xr, js:je, ls:le], axis=0)
    m[xr - 1, js:je, ls:le] = m[0, js:je, ls:le]
    m[imr - xr + 1, js:je, ls:le] = seqsum(m[imr - xr + 1 : imr, js:je, ls:le], axis=0)
    m[imr - xr + 2, js:je, ls:le] = m[imr + 1, js:je, ls:le]
    if not tracers:
        return
    rm = f["rm"]
    rm[xr, js:je, ls:le] = seqsum(rm[1:xr, js:je, ls:le], axis=0)
    rm[imr - xr + 1, js:je, ls:le] = seqsum(rm[imr - xr + 1 : imr, js:je, ls:le], axis=0)
    for name in ("rm", "rxm", "rym", "rzm"):
        q = f[name]
        q[xr - 1, js:je, ls:le] = q[0, js:je, ls:le]
        q[imr - xr + 2, js:je, ls:le] = q[imr + 1, js:je, ls:le]


def _uncompress_xedges(zoom, st, region, js, je, ls, le, tracers=True):
    xr, g = zoom.ref(region, "x"), zoom.g(region)
    if xr == 1 or g.im // g.xref == zoom.g(1).im:
        return
    f, imr = st[region], g.im
    m = f["m"]
    for lo, hi, edge in ((1, xr, xr), (imr - xr + 1, imr, imr - xr + 1)):
        m_edge = m[edge, js:je, ls:le].copy()
        m[lo:hi, js:je, ls:le] = (m_edge / xr)[None]
        for name in ("rm", "rxm", "rym", "rzm") if tracers else ():
            q = f[name]
            q_edge = q[edge, js:je, ls:le].copy()
            q[lo:hi, js:je, ls:le] = (q_edge / xr)[None]


def _compress_yedges(zoom, st, region, is_, ie, ls, le, tracers=True):
    yr, g = zoom.ref(region, "y"), zoom.g(region)
    if yr == 1 or region == 1:
        return
    f, jmr = st[region], g.jm
    m = f["m"]
    m[is_:ie, yr, ls:le] = seqsum(m[is_:ie, 1:yr, ls:le], axis=1)
    m[is_:ie, yr - 1, ls:le] = m[is_:ie, 0, ls:le]
    m[is_:ie, jmr - yr + 1, ls:le] = seqsum(m[is_:ie, jmr - yr + 1 : jmr, ls:le], axis=1)
    m[is_:ie, jmr - yr + 2, ls:le] = m[is_:ie, jmr + 1, ls:le]
    if not tracers:
        return
    rm = f["rm"]
    rm[is_:ie, yr, ls:le] = seqsum(rm[is_:ie, 1:yr, ls:le], axis=1)
    rm[is_:ie, jmr - yr + 1, ls:le] = seqsum(rm[is_:ie, jmr - yr + 1 : jmr, ls:le], axis=1)
    for name in ("rm", "rxm", "rym", "rzm"):
        q = f[name]
        q[is_:ie, yr - 1, ls:le] = q[is_:ie, 0, ls:le]
        q[is_:ie, jmr - yr + 2, ls:le] = q[is_:ie, jmr + 1, ls:le]


def _uncompress_yedges(zoom, st, region, is_, ie, ls, le, tracers=True):
    yr, g = zoom.ref(region, "y"), zoom.g(region)
    if yr == 1 or region == 1:
        return
    f, jmr = st[region], g.jm
    for lo, hi, edge in ((1, yr, yr), (jmr - yr + 1, jmr, jmr - yr + 1)):
        for name in ("m", "rm", "rxm", "rym", "rzm") if tracers else ("m",):
            q = f[name]
            q_edge = q[is_:ie, edge, ls:le].copy()
            q[is_:ie, lo:hi, ls:le] = (q_edge / yr)[:, None]


# ------------------------------------------------------------------ mix_edges (zoom_tools)
def mix_edges(zoom, st, region, tracers=True):
    """zoom_tools.F90 mix_edges: every parent-cell block of the child's `edge == -1` ring
    replaced by its mean -- m, rm and the three slopes.  tracers=False is advectm_cfl.F90's
    mixm_edges, m alone."""
    g = zoom.g(region)
    xr, yr, zr = zoom.ref(region, "x"), zoom.ref(region, "y"), zoom.ref(region, "z")
    if zr != 1:
        raise ValueError("mix_edges: zooming in z direction not supported!")
    xedges, ysp, ynp = True, True, True
    if xr == 1 or g.im // g.xref == zoom.g(1).im:
        xedges = False
        if region == 1:
            return
    if g.touch_sp == 1:
        ysp = False
    if g.touch_np == 1:
        ynp = False
    xyref = xr * yr
    imr, jmr = g.im, g.jm
    imp, jmp = imr // xr, jmr // yr
    ring = zoom.edge[region] == -1  # (im, jm)

    def block_means(q):
        """to_parent(ip, jp) over the ring, accumulated j outer / i inner as the Fortran loops."""
        blk = np.where(ring, q / xyref, 0.0).reshape(imp, xr, jmp, yr)
        acc = np.zeros((imp, jmp))
        for b in range(yr):
            for a in range(xr):
                acc = acc + blk[:, a, :, b]
        return acc

    def spread(q, to_parent):
        ip = np.arange(imr) // xr
        if xedges:
            jp = np.arange(jmr) // yr
            q[:xr, :] = to_parent[0, jp][None, :]
            q[imr - xr :, :] = to_parent[imp - 1, jp][None, :]
        if ysp:
            q[:, :yr] = to_parent[ip, 0][:, None]
        if ynp:
            q[:, jmr - yr :] = to_parent[ip, jmp - 1][:, None]

    f = st[region]
    for lev in range(1, zoom.lm + 1):
        q = f["m"][1:imr, 1:jmr, lev]
        spread(q, block_means(q))
    for n in range(1, st.nt + 1 if tracers else 1):
        for lev in range(1, zoom.lm + 1):
            qs = [f[name][1:imr, 1:jmr, lev, n] for name in ("rm", "rxm", "rym", "rzm")]
            means = [block_means(q) for q in qs]  # all four accumulated before any is replaced
            for q, t in zip(qs, means, strict=True):
                spread(q, t)


# ------------------------------------------------------------------ update_parent (zoom_tools)
def update_parent(zoom, st, region, tracers=True):
    """zoom_tools.F90 update_parent: m and rm over the child footprint replaced by the child's
    block sums (accumulated i outer, j inner).  The parent's slopes are left as they were.
    tracers=False is advectm_cfl.F90's updatem_parent."""
    if region == 1:
        return
    g, parent = zoom.g(region), zoom.g(region).parent
    xr, yr, zr = zoom.ref(region, "x"), zoom.ref(region, "y"), zoom.ref(region, "z")
    imp, jmp = g.im // xr, g.jm // yr
    ib, ie, jb, je = zoom.ibeg[region], zoom.iend[region], zoom.jbeg[region], zoom.jend[region]
    f, p = st[region], st[parent]

    def block_sums(q):  # (im, jm, lm) -> (imp, jmp, lm)
        blk = q.reshape(imp, xr, jmp, yr, *q.shape[2:])
        acc = np.zeros((imp, jmp, *q.shape[2:]))
        for a in range(xr):
            for b in range(yr):
                acc = acc + blk[:, a, :, b]
        return acc

    lm = zoom.lm
    p["m"][ib:ie, jb:je, 1:lm] = block_sums(f["m"][1 : g.im, 1 : g.jm, 1:lm])
    if zr != 1:
        raise ValueError("update_parent: zooming in z direction not supported")
    for n in range(1, st.nt + 1 if tracers else 1):
        p["rm"][ib:ie, jb:je, 1:lm, n] = block_sums(f["rm"][1 : g.im, 1 : g.jm, 1:lm, n])
