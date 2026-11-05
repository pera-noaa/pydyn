"""convec on Metal (and CUDA) for one lmax n <= 32, as unrolled kernels: the source and the arguments.

The Metal compiler keeps loops over small arrays rolled (and the arrays in memory) whatever
the pragma, and one kernel holding all of csrc/convec.cuh's pd_convec is too large to run well
unrolled, so pd_convec is written out here per n, every array element a named scalar, in four
passes through device scratch (A, Y: n x n per column, row-major):

    pd_cvb_<n>  the build (pd_convection_matrix): a SIMD group a column, lane p holding level p
                of the inputs (broadcast to every lane) and column p of FU and FD -> A
    pd_cvl_<n>  the LU (pd_crout's first loops): lane r holding row r; right-looking, which gives
                each element its updates in ascending k and its division last, as pd_crout's
                left-looking loops do, the pivot row broadcast from its lane -> A
    pd_cvs_<n>  the inverse (pd_crout's sweeps): a thread per column of y -> Y
    pd_cvu_<n>  the update: the slopes by the diagonal, rm(l) = sum_k C(l, k) rm(k) in k order,
                and the column's before-minus-after on its largest level (convec.cuh pd_column_deficit)

and first pd_cvc_<n>, which compares each column's inputs (m, en/detrainment, dkg, the cloud
levels) bitwise with the copy kept from the last call on these columns: where they are as they
were, the three passes before the update return and the update uses the inverse kept -- the
same bits, computed once.  (A step's two convec calls on region 1 see the same mass: only
sources and chemistry run between them.)

Every element's operations are pd_convec's, in its order; tests/test_metal.py holds the four
bitwise to the host build of convec.cuh.
"""

import re

__all__ = ["ARGS", "convec_source"]

_DIALECT = "metal"  # convec_source's: "metal" (MSL, float) or "cuda" (CUDA C++, csrc's real: double)

# CUDA / HIP: the SIMD-group intrinsics the passes use, as pd.h's 32-lane group operations
_CUDA_PRELUDE = """
typedef unsigned int uint;
typedef unsigned short ushort;
#define simd_broadcast(v, l) pd_shfl((v), (l))
#define simd_shuffle_up(v, d) pd_shfl_up((v), (d))
#define simd_any(b) pd_any((int)(b))
"""

# the argument structs, (ctype, name) as backend.kernel_signatures gives them
_COLS = [("const int*", "cols"), ("i64", "jm"), ("i64", "ncol")]
ARGS = {
    "cvb": [
        *_COLS,
        ("real", "dt"),
        ("const real*", "m"),
        ("i64", "sm0"),
        ("i64", "sm1"),
        ("i64", "sm2"),
        ("const real*", "entu"),
        ("const real*", "detu"),
        ("const real*", "entd"),
        ("const real*", "detd"),
        ("i64", "se0"),
        ("i64", "se1"),
        ("i64", "se2"),
        ("const real*", "dkg"),
        ("i64", "sd0"),
        ("i64", "sd1"),
        ("i64", "sd2"),
        ("const int*", "ctop"),
        ("const int*", "clfs"),
        ("i64", "sc0"),
        ("i64", "sc1"),
        ("real*", "A"),
        ("const int*", "F"),
    ],
    "cvl": [("i64", "ncol"), ("real*", "A"), ("const int*", "F")],
    "cvg": [("i64", "ncol"), ("const real*", "A"), ("real*", "Y"), ("const int*", "F"), ("int*", "info")],
    "cvs": [("i64", "ncol"), ("const real*", "A"), ("real*", "Y"), ("const int*", "F")],
    "cvc": [
        *_COLS,
        ("const real*", "m"),
        ("i64", "sm0"),
        ("i64", "sm1"),
        ("i64", "sm2"),
        ("const real*", "entu"),
        ("const real*", "detu"),
        ("const real*", "entd"),
        ("const real*", "detd"),
        ("i64", "se0"),
        ("i64", "se1"),
        ("i64", "se2"),
        ("const real*", "dkg"),
        ("i64", "sd0"),
        ("i64", "sd1"),
        ("i64", "sd2"),
        ("const int*", "ctop"),
        ("const int*", "clfs"),
        ("i64", "sc0"),
        ("i64", "sc1"),
        ("real*", "S"),
        ("int*", "F"),
    ],
    "cvu": [
        *_COLS,
        ("i64", "nt"),
        ("const real*", "Y"),
        ("real*", "rm"),
        ("real*", "rxm"),
        ("real*", "rym"),
        ("real*", "rzm"),
        ("i64", "sq0"),
        ("i64", "sq1"),
        ("i64", "sq2"),
        ("i64", "sq3"),
    ],
}
# CUDA's pd_cvf_<n>: compare, build, LU, sweeps and update in one launch, the matrix in shared memory
ARGS["cvf"] = [
    *ARGS["cvc"][:-2],  # the columns and every input compared and built from
    ("real", "dt"),
    ("real*", "S"),
    ("real*", "Y"),
    ARGS["cvu"][3],  # nt
    *ARGS["cvu"][5:],  # rm, rxm, rym, rzm and their strides
]
CVF_WARPS = 4  # warps a block: backend._THREADS / 32, the shared memory's division
_C = {"i64": "long", "int": "int", "real": "real"}


def _struct(name, n):
    fields = []
    for ctype, f in ARGS[name]:
        if _DIALECT == "cuda":
            fields.append(f"    {ctype[:-1]} *{f};" if ctype.endswith("*") else f"    {ctype} {f};")
        else:
            fields.append(
                f"    device {ctype[:-1]} *{f};" if ctype.endswith("*") else f"    {_C[ctype]} {f};"
            )
    return f"struct pd_{name}_{n}_args {{\n" + "\n".join(fields) + "\n};\n"


def _head(name, n, per_lane, skip=False):
    """The kernel's opening: q the column (a SIMD group's, or 32 threads' when per_lane);
    skip: return where the compare pass found the column's inputs as they were."""
    if _DIALECT == "cuda":  # the arguments as CuPy passes them, gathered into the same struct
        params = ", ".join(f"{c[:-1]} *a_{f}" if c.endswith("*") else f"{c} a_{f}" for c, f in ARGS[name])
        opening = [
            f'extern "C" __global__ void pd_{name}_{n}({params})',
            "{",
            f"    const pd_{name}_{n}_args a = {{{', '.join(f'a_{f}' for _, f in ARGS[name])}}};",
            "    uint t = blockIdx.x * blockDim.x + threadIdx.x;",
        ]
    else:
        opening = [
            f"kernel void pd_{name}_{n}(constant pd_{name}_{n}_args &a [[buffer(0)]],",
            "    uint t [[thread_position_in_grid]])",
            "{",
        ]
    return [
        *opening,
        "    i64 q = (i64)(t / 32);",
        "    const int p = (int)(t % 32);",
        "    if (q >= a.ncol) return;" + ("" if per_lane else " /* uniform over the SIMD group */"),
        *(["    if (!a.F[q]) return; /* the inputs as they were: A and Y kept */"] if skip else []),
        f"    const bool own = p < {n};",
    ]


def _bc(v, lane):
    return f"simd_broadcast({v}, (ushort){lane})"


def _compare(n):
    """Each column's inputs against the copy kept of them, bitwise; F[q] = 1 where any
    differs (or nothing was kept: the copy starts as NaN), and the copy updated."""
    o = _head("cvc", n, False)
    w = o.append
    _compare_body(w, n)
    w("    if (p == 0) a.F[q] = changed ? 1 : 0;")
    w("    if (!changed) return;")
    _compare_store(w, n)
    w("}")
    return o


def _bits():
    return (
        ("u64", "(u64)__double_as_longlong", "__double_as_longlong", "__longlong_as_double")
        if _DIALECT == "cuda"
        else ("uint", "as_type<uint>", "as_type<int>", "as_type<real>")
    )


_COMPARED = 6  # m, entu, detu, entd, detd, dkg: v0..v5, the copy's rows 0..5


def _compare_body(w, n):
    """The column's inputs into v0..v5, ct and cl, and `bool changed` (any bit differs)."""
    w("    i64 col = a.cols[q], i = col / a.jm, j = col % a.jm;")
    w("    i64 oe = i * a.se0 + j * a.se1;")
    w(f"    device real *S = a.S + q * {6 * n + 2};")
    word, bits, ibits, _ = _bits()
    w(f"    {word} diff = 0u;")
    w("    real v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f, v5 = 0.0f;")
    w("    if (own) {")
    vals = [
        ("a.m[i * a.sm0 + j * a.sm1 + p * a.sm2]", 0),
        ("a.entu[oe + p * a.se2]", 1),
        ("a.detu[oe + p * a.se2]", 2),
        ("a.entd[oe + p * a.se2]", 3),
        ("a.detd[oe + p * a.se2]", 4),
        ("a.dkg[i * a.sd0 + j * a.sd1 + p * a.sd2]", 5),
    ]
    for expr, k in vals:
        w(f"        v{k} = {expr};")
        w(f"        diff |= {bits}(v{k}) ^ {bits}(S[{k * n} + p]);")
    w("    }")
    w("    int ct = a.ctop[i * a.sc0 + j * a.sc1], cl = a.clfs[i * a.sc0 + j * a.sc1];")
    w(f"    if (p == 0) diff |= ({word})(ct ^ {ibits}(S[{6 * n}]));")
    w(f"    if (p == 0) diff |= ({word})(cl ^ {ibits}(S[{6 * n + 1}]));")
    w("    bool changed = simd_any(diff != 0u);")


def _compare_store(w, n):
    """The copy updated from _compare_body's values."""
    frombits = _bits()[3]
    w("    if (own) {")
    for k in range(_COMPARED):
        w(f"        S[{k * n} + p] = v{k};")
    w("    }")
    cast = "(long long)" if _DIALECT == "cuda" else ""
    w(f"    if (p == 0) {{ S[{6 * n}] = {frombits}({cast}ct); S[{6 * n + 1}] = {frombits}({cast}cl); }}")


def _build(n):
    o = _head("cvb", n, False, skip=True)
    w = o.append
    _build_body(w, n)
    w("    if (!own) return;")
    _build_tail(w, n)
    w(f"    device real *A = a.A + q * {n * n} + p;")
    for k in range(n):  # A(k, p) = FD(k, p): row-major, as CM(a, n, r, c) is FD(r, c)
        w(f"    A[{k * n}] = fd{k};")
    w("}")
    return o


def _build_body(w, n):
    """pd_convection_matrix's inputs and its two recurrences: lane p's FU and FD columns."""
    w("    i64 col = a.cols[q], i = col / a.jm, j = col % a.jm;")
    w("    i64 oe = i * a.se0 + j * a.se1;")
    w("    int ctop = a.ctop[i * a.sc0 + j * a.sc1], clfs = a.clfs[i * a.sc0 + j * a.sc1];")
    w("    real dt = a.dt;")
    w("    real in_eu = 0.0f, in_du = 0.0f, in_ed = 0.0f, in_dd = 0.0f, in_m = 1.0f, in_dg = 0.0f;")
    w("    if (own) {")
    for name, arr in (("eu", "entu"), ("du", "detu"), ("ed", "entd"), ("dd", "detd")):
        w(f"        in_{name} = a.{arr}[oe + p * a.se2];")
    w("        in_m = a.m[i * a.sm0 + j * a.sm1 + p * a.sm2];")
    w("        in_dg = a.dkg[i * a.sd0 + j * a.sd1 + p * a.sd2];")
    w("    }")
    # the divisions only lane p's own row needs, with its own level's operands, once
    w("    real eu_m = in_eu / in_m, ned_m = -in_ed / in_m; /* eu / m(k-1), -ed / m(k-1) at p = k-1 */")
    w("    real dg_m = in_dg / in_m;                         /* dkg(k-1) / m(k-1) at p = k-1 */")
    w("    real dgp_m = simd_shuffle_up(in_dg, (ushort)1) / in_m; /* dkg(k-1) / m(k) at p = k */")
    w("    real " + ", ".join(f"fu{k} = 0.0f, fd{k} = 0.0f" for k in range(n + 1)) + ";")
    w("    real amu = 0.0f, my_amu = 0.0f, amd = 0.0f, my_amd = 0.0f; /* my_amu: amu(p); my_amd: amd(p+1) */")

    def updraft(k):  # FU(k, p) from FU(k-1, p), this lane's
        w("    {")
        w(f"        int act = ctop >= {k};")
        w(f"        real eu = {_bc('in_eu', k - 1)}, du = {_bc('in_du', k - 1)};")
        w("        real av = amu + eu - du;")
        w("        int up = av > 0.0f;")
        w("        real zxi = up ? np_maximum(0.0f, 1.0f - du / (amu + eu)) : 0.0f;")
        w("        amu = act && up ? av : 0.0f;")
        w(f"        if (p == {k}) my_amu = amu;")
        w(f"        if (p < {k - 1}) fu{k} = act ? fu{k - 1} * zxi : 0.0f;")
        w(f"        if (p == {k - 1}) fu{k} = act ? eu_m * zxi : 0.0f;")
        w("    }")

    def downdraft(k):  # FD(k-1, p) from FD(k, p)
        w("    {")
        w(f"        int act = clfs >= {k};")
        w(f"        real ed = {_bc('in_ed', k - 1)}, dd = {_bc('in_dd', k - 1)};")
        w("        real av = amd - ed + dd;")
        w("        int dn = av < 0.0f;")
        w("        real zxi = dn ? np_maximum(0.0f, 1.0f + dd / (amd - ed)) : 0.0f;")
        w("        amd = act && dn ? av : 0.0f;")
        w(f"        if (p == {k - 2}) my_amd = amd;")
        w(f"        if (p >= {k} && own) fd{k - 1} = act ? fd{k} * zxi : fd{k - 1};")
        w(f"        if (p == {k - 1}) fd{k - 1} = act ? ned_m * zxi : fd{k - 1};")
        w("    }")

    # the two recurrences share nothing (FU and FD, amu and amd): interleaved, each in its order
    ups, downs = list(range(1, n + 1)), list(range(n, 1, -1))
    for s in range(max(len(ups), len(downs))):
        if s < len(ups):
            updraft(ups[s])
        if s < len(downs):
            downdraft(downs[s])


def _build_tail(w, n):
    """The combine on an owning lane: fd0..fd(n-1) = A(k, p)."""
    w("    real amu_m = my_amu / in_m; /* amu(k) / m(k) at p = k */")
    w("    real amd_m = my_amd / in_m; /* amd(k) / m(k-1) at p = k-1 */")
    for k in range(1, n):  # combine, compensating subsidence, diffusion
        w(f"    fd{k} = fu{k} + fd{k};")
        w(f"    if (p == {k}) fd{k} = fd{k} - amu_m;")
        w(f"    if (p == {k - 1}) {{")
        w(f"        fd{k} = fd{k} - amd_m;")
        w(f"        fd{k} = fd{k} + dg_m;")
        w("    }")
        w(f"    if (p == {k}) fd{k} = fd{k} - dgp_m;")
    for k in range(1, n + 1):  # conv1: row k-1 from rows k-1 and k
        w(f"    fd{k - 1} = -dt * (fd{k - 1} - fd{k});")
        w(f"    if (p == {k - 1}) fd{k - 1} = fd{k - 1} + 1.0f;")


def _lu(n):
    o = _head("cvl", n, False, skip=True)
    w = o.append
    w(f"    device real *A = a.A + q * {n * n} + p * {n};")
    w("    real " + ", ".join(f"ar{c} = own ? A[{c}] : 0.0f" for c in range(n)) + ";")
    _lu_body(w, n)
    w("    if (!own) return;")
    for c in range(n):
        w(f"    A[{c}] = ar{c};")
    w("}")
    return o


def _lu_body(w, n):
    """pd_crout's elimination on lane p's row ar0..ar(n-1), right-looking."""
    for k in range(n):
        w("    {")
        w(f"        real piv = {_bc(f'ar{k}', k)};")
        w(f"        bool low = own && p > {k};")
        w(f"        if (low) ar{k} = ar{k} / piv;")
        for c in range(k + 1, n):
            w(f"        {{ real u = {_bc(f'ar{c}', k)}; if (low) ar{c} = ar{c} - ar{k} * u; }}")
        w("    }")


def _dtrsm_rows(n):
    """split[(k, c)]: in DGETRF2's recursion the update of column c by step k (k < c) is a
    DTRSM on the rows below k and above split, a DGEMM from split down (the node whose
    left half holds k and right half c: its n1 boundary).  DTRSM skips u(k, c) == 0, DGEMM
    does not: the same c - u * l otherwise (DGEMM's c + (-1 * u) * l is that, exactly)."""
    split = {}

    def node(lo, nc):  # rows lo..n-1, columns lo..lo+nc-1
        if nc <= 1 or n - lo <= 1:
            return
        n1 = min(n - lo, nc) // 2
        for k in range(lo, lo + n1):
            for c in range(lo + n1, lo + nc):
                split[(k, c)] = lo + n1
        node(lo, n1)
        node(lo + n1, nc - n1)

    node(0, n)
    return split


def _getrf_getri(n):
    """The LAPACK kind: csrc/convec.cuh's pd_dgetrf2 + pd_dgetri on lane p's row p of conv1,
    every element given the serial twin's operations in its order -- DGETRF2's leaves as a
    right-looking LU (pivot by IDAMAX's first largest |x|, rows swapped whole, the column
    scaled by 1 / pivot, or divided below SFMIN), DTRTI2's columns, DGETRI's sweep from the
    last column, its column interchanges -- the inverse's row p into Y."""
    split = _dtrsm_rows(n)
    o = _head("cvg", n, False, skip=True)
    w = o.append
    w(f"    device const real *A = a.A + q * {n * n} + (own ? p : 0) * {n};")
    w("    real " + ", ".join(f"ar{c} = own ? A[{c}] : 0.0f" for c in range(n)) + ";")
    w("    int bad = 0;")
    w("    int " + ", ".join(f"pv{k} = {k}" for k in range(n)) + ";")
    for k in range(n):
        w("    {")
        if k < n - 1:  # DGETRF2's n == 1 leaf on rows k..n-1 (the last is its m == 1 leaf)
            w(f"        int im = {k};")
            w(f"        real dmax = fabs({_bc(f'ar{k}', k)});")
            for r in range(k + 1, n):
                w(f"        {{ real v = fabs({_bc(f'ar{k}', r)}); if (v > dmax) {{ im = {r}; dmax = v; }} }}")
            w(f"        pv{k} = im;")
            w(f"        real piv = simd_broadcast(ar{k}, (ushort)im);")
            w("        if (piv != 0.0f) {")
            w(f"            if (im != {k}) {{ /* DLASWP: the rows swapped whole (only moves) */")
            for c in range(n):
                w(f"                {{ real x = {_bc(f'ar{c}', k)}, y = simd_broadcast(ar{c}, (ushort)im);")
                w(f"                  if (p == {k}) ar{c} = y; if (p == im) ar{c} = x; }}")
            w("            }")
            w(f"            bool low = own && p > {k};")
            w("            if (fabs(piv) >= PD_REAL_TINY) {")
            w("                real r = 1.0f / piv;")
            w(f"                if (low) ar{k} = r * ar{k};")
            w("            } else {")
            w(f"                if (low) ar{k} = ar{k} / piv;")
            w("            }")
            w("        } else {")
            w("            bad = 1;")
            w("        }")
        else:
            w(f"        if ({_bc(f'ar{k}', k)} == 0.0f) bad = 1;")
        for c in range(k + 1, n):  # DTRSM above split (skips u == 0), DGEMM from split
            cut = split[(k, c)]
            w(f"        {{ real u = {_bc(f'ar{c}', k)};")
            w(f"          if (own && p > {k} && !(p < {cut} && u == 0.0f)) ar{c} = ar{c} - u * ar{k}; }}")
        w("    }")
    # DGETRI: a zero on U's diagonal is DGETRI's info (the result is not used then)
    w("    { bool z = own && (" + " || ".join(f"(p == {i} && ar{i} == 0.0f)" for i in range(n)) + ");")
    w("      if (simd_any(z)) bad = 1; }")
    for j in range(n):  # DTRTI2, column j: x = U(0..j-1, j), DTRMV with the inverted columns < j, DSCAL
        w("    {")
        w(f"        if (p == {j}) ar{j} = 1.0f / ar{j};")
        w(f"        real ajj = -{_bc(f'ar{j}', j)};")
        for jj in range(j):
            w(f"        {{ real temp = {_bc(f'ar{j}', jj)};")
            w(f"          if (temp != 0.0f) {{ if (p < {jj}) ar{j} = ar{j} + temp * ar{jj};")
            w(f"                               if (p == {jj}) ar{j} = ar{j} * ar{jj}; }} }}")
        w(f"        if (p < {j}) ar{j} = ajj * ar{j};")
        w("    }")
    for j in range(n - 1, -1, -1):  # the solve inv(A) L = inv(U): work = L(j+1.., j), then DGEMV
        w("    {")
        cols = list(range(j + 1, n))
        if cols:
            w("        real " + ", ".join(f"w{c} = {_bc(f'ar{j}', c)}" for c in cols) + ";")
            w(f"        if (p > {j}) ar{j} = 0.0f;")
            for c in cols:
                w(f"        {{ real temp = -1.0f * w{c}; ar{j} = ar{j} + temp * ar{c}; }}")
        w("    }")
    for j in range(n - 2, -1, -1):  # the column interchanges, in DGETRI's order (moves only)
        w(f"    if (pv{j} != {j}) {{")
        for c in range(j + 1, n):
            w(f"        if (pv{j} == {c}) {{ real x = ar{j}; ar{j} = ar{c}; ar{c} = x; }}")
        w("    }")
    w("    if (bad && p == 0) a.info[0] = 1;")
    w("    if (!own) return;")
    w(f"    device real *Y = a.Y + q * {n * n} + p * {n};")
    for c in range(n):
        w(f"    Y[{c}] = ar{c};")
    w("}")
    return o


def _sweeps(n):
    o = _head("cvs", n, True, skip=True)
    w = o.append
    w("    if (!own) return;")
    w(f"    device const real *A = a.A + q * {n * n};")
    w("    real " + ", ".join(f"y{i} = p == {i} ? 1.0f : 0.0f" for i in range(n)) + ";")
    _sweeps_body(w, n, lambda i, j: f"A[{i * n + j}]")
    w(f"    device real *Y = a.Y + q * {n * n} + p;")
    for i in range(n):
        w(f"    Y[{i * n}] = y{i};")
    w("}")
    return o


def _sweeps_body(w, n, at):
    """pd_crout's sweeps: y = column p of the inverse, from the LU's elements at(i, j)."""
    for i in range(1, n):  # forward: y(i, c) -= A(i, j) y(j, c), j ascending
        for j in range(i):
            w(f"    y{i} = y{i} - {at(i, j)} * y{j};")
    for i in range(n - 1, -1, -1):  # back: j ascending, then the division
        for j in range(i + 1, n):
            w(f"    y{i} = y{i} - {at(i, j)} * y{j};")
        w(f"    y{i} = y{i} / {at(i, i)};")


def _update(n):
    o = _head("cvu", n, False)
    w = o.append
    w("    i64 col = a.cols[q], i = col / a.jm, j = col % a.jm;")
    w("    i64 base = i * a.sq0 + j * a.sq1 + (own ? p : 0) * a.sq2;")
    w(f"    device const real *Y = a.Y + q * {n * n} + (own ? p : 0) * {n};")
    _update_body(w, n)
    w("}")
    return o


def _update_body(w, n):
    """The slopes by the diagonal and rm = C rm, every tracer, from Y: lane p's row."""
    w("    real c_ll = own ? Y[own ? p : 0] : 0.0f;")
    w("    for (i64 t = 0; t < a.nt; t++) {")
    w("        i64 o = base + t * a.sq3;")
    w("        if (own) {")
    w("            a.rxm[o] = c_ll * a.rxm[o];")
    w("            a.rym[o] = c_ll * a.rym[o];")
    w("            a.rzm[o] = c_ll * a.rzm[o];")
    w("        }")
    w("        real rk = own ? a.rm[o] : 0.0f;")
    w("        real z = 0.0f;")
    for k in range(n):
        w(f"        z = z + Y[{k}] * {_bc('rk', k)};")
    if _DIALECT == "cuda":  # double: no deficit (convec.cuh's is PD_F32's)
        w("        if (own) a.rm[o] = z;")
        w("    }")
        return
    w("        real sb = 0.0f, cb = 0.0f, sa = 0.0f, ca = 0.0f, x, u, bp; /* convec.cuh pd_column_deficit */")
    w("        int lx = 0;")
    w("        real zx = 0.0f;")
    for k in range(n):
        w(f"        x = {_bc('rk', k)};")
        w("        u = sb + x; bp = u - sb; cb = cb + ((sb - (u - bp)) + (x - bp)); sb = u;")
    for k in range(n):
        w(f"        x = {_bc('z', k)};")
        w("        u = sa + x; bp = u - sa; ca = ca + ((sa - (u - bp)) + (x - bp)); sa = u;")
        w("        zx = x;" if k == 0 else f"        if (x > zx) {{ zx = x; lx = {k}; }}")
    w("        if (own) a.rm[o] = p == lx ? z + ((sb - sa) + (cb - ca)) : z;")
    w("    }")


def _fused(n):
    """CUDA: pd_convec's Crout kind for a warp's column in one kernel, every element given the
    passes' operations in their order; the matrix moves between them through the warp's
    shared memory (rows padded to an odd pitch: no bank conflicts), not through A in device
    memory, and Y is written once, kept for a call whose inputs are as they were."""
    pitch = n | 1
    o = _head("cvf", n, False)
    w = o.append
    w(f"    __shared__ real pd_cvf_sh[{CVF_WARPS * n * pitch}];")
    w(f"    real *sh = pd_cvf_sh + (threadIdx.x / 32) * {n * pitch};")
    w(f"    real *Yk = a.Y + q * {n * n} + p; /* the inverse kept: Yk[i * n] = Y(i, p) */")
    w("    bool fresh;")
    w("    { /* pd_cvc */")
    _compare_body(w, n)
    w("    fresh = changed;")
    w("    if (changed) {")
    _compare_store(w, n)
    w("    } }")
    w("    if (fresh) {")
    w("    { /* pd_cvb: lane p's column of A into sh */")
    _build_body(w, n)
    w("    if (own) {")
    _build_tail(w, n)
    for k in range(n):
        w(f"    sh[{k * pitch} + p] = fd{k};")
    w("    } }")
    w("    pd_group_sync();")
    w("    { /* pd_cvl: lane p's row */")
    w("    real " + ", ".join(f"ar{c} = own ? sh[p * {pitch} + {c}] : 0.0f" for c in range(n)) + ";")
    _lu_body(w, n)
    w("    if (own) {")
    for c in range(n):
        w(f"    sh[p * {pitch} + {c}] = ar{c};")
    w("    } }")
    w("    pd_group_sync();")
    w("    { /* pd_cvs: lane p's column of the inverse */")
    w("    real " + ", ".join(f"y{i} = p == {i} ? 1.0f : 0.0f" for i in range(n)) + ";")
    w("    if (own) {")
    _sweeps_body(w, n, lambda i, j: f"sh[{i * pitch + j}]")
    w("    }")
    w("    pd_group_sync();")
    w("    if (own) {")
    for i in range(n):
        w(f"    sh[{i * pitch} + p] = y{i}; Yk[{i * n}] = y{i};")
    w("    } }")
    w("    } else if (own) {")
    for i in range(n):
        w(f"    sh[{i * pitch} + p] = Yk[{i * n}];")
    w("    }")
    w("    pd_group_sync();")
    w("    { /* pd_cvu, lane p's row of the inverse */")
    w("    i64 col = a.cols[q], i = col / a.jm, j = col % a.jm;")
    w("    i64 base = i * a.sq0 + j * a.sq1 + (own ? p : 0) * a.sq2;")
    w(f"    const real *Y = sh + (own ? p : 0) * {pitch};")
    _update_body(w, n)
    w("    }")
    w("}")
    return o


def convec_source(n, dialect="metal"):
    """The passes for lmax n with their argument structs: MSL (float), or CUDA C++ in csrc's
    double (the same program: the warp a SIMD group, its literals double, no float deficit)."""
    global _DIALECT
    if not 1 <= n <= 32:
        raise ValueError(f"convec_gen: n = {n} (1..32)")
    if dialect not in ("metal", "cuda"):
        raise ValueError(f"convec_gen: dialect {dialect!r}")
    _DIALECT = dialect
    try:
        cuda = dialect == "cuda"
        parts = [_struct(name, n) for name in ARGS if cuda or name != "cvf"]
        for gen in (_compare, _build, _lu, _sweeps, _getrf_getri, _update, *([_fused] if cuda else [])):
            parts.append("\n".join(gen(n)) + "\n")
    finally:
        _DIALECT = "metal"
    text = "\n".join(parts)
    if dialect == "cuda":
        text = re.sub(r"\bdevice ", "", text)
        text = re.sub(r"(?<![\w.])(\d+\.\d+)f\b", r"\1", text)  # 1.0f -> 1.0: double literals
        text = _CUDA_PRELUDE + text
    return text
