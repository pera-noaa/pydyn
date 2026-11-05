"""Every numeric constant the transcription mirrors, pinned to its Fortran source.

Every other test here compares *arrays*: give both sides the same inputs, demand the same
outputs.  That leaves two ways a constant can be wrong and the suite still pass.

  never exercised   the value only decides behaviour at the margin and no observed run
                    reaches the margin.  `Check_CFL` picks an `ndyn` at which y and z never
                    sub-cycle, so `cfl_max_nloop_{y,z}` could hold any value at all
                    (`docs/cfl_headroom.md`).  Sub-cycle caps, iteration limits and floors
                    are all like this.
  only exercised     the physical constants are covered by the §3b replay -- but only where
  with the dumps     8.3 GB of dump sets is on disk.  Here they are checked from the source
                     alone, in 0.1 s, anywhere a tm5-zoom checkout exists.

Not hypothetical: `cfl_max_nloop_{y,z}` went 6, 10 -> 1, 1 in tm5-zoom `6f59f9f72`
(2026-04-19), inside a commit whose subject is an `MPI_Allreduce` fix.  A transcription
whose constants have silently drifted is worse than no transcription, because it still
passes.

Three ways this file fails, and all three are the point:

  value differs     the model moved a constant; decide what the transcription should now
                    say, and whether a claim in `docs/state.md` rested on the old value
  name not found    the declaration was restructured, renamed or moved; the mirror is
                    unverifiable until someone re-points it, which is a finding
  not accounted for a new module-level numeric constant appeared in `pydyn/` and is in
                    neither `CONSTANTS` nor `WAIVED`, so nobody has said where it comes from

The tests that read the Fortran skip without a checkout; the accounting test and the device
cap's (`csrc/pd.h`) need none and run everywhere, CI included.  The conftest header says
whether the checkout is at the pin (fortran/TM5ZOOM_PIN).  `PYD-12`.
"""

import ast
import importlib
import pkgutil
import re
from functools import cache

import pytest

import pydyn

from common import TM5ZOOM

needs_checkout = pytest.mark.skipif(
    not TM5ZOOM.is_dir(), reason=f"no tm5-zoom checkout at {TM5ZOOM} (set PYDYN_TM5ZOOM)"
)

ADVM = "proj/ct-unified/trunk/src/advectm_cfl.F90"
ADVX = "proj/ct-unified/trunk/src/advectx__slopes.F90"
ADVY = "proj/ct-unified/trunk/src/advecty__slopes.F90"
ADVZ = "proj/ct-unified/branches/speedup/src/advectz__slopes.F90"
BINAS = "base/trunk/src/binas.F90"
CHEM = "proj/ct-unified/branches/chem_unified/src/chemistry_ch4.F90"
CHEMP = "proj/ct-unified/trunk/src/chem_param.F90"
DIFFK = "proj/ct-unified/trunk/src/diffusion_kernels.F90"
DIMS = "base/trunk/src/dims.F90"
LEVELS = "proj/levels/ml137/tropo34/src/dims_levels__ml137_tropo34.F90"
READOH = "proj/ct-unified/branches/chem_unified/src/read_OH_field.F90"
REGRID = "proj/ct-unified/trunk/src/emission_regrid.F90"
SOURCES = "proj/ct-unified/trunk/src/sources_sinks__ctuni.F90"
METEO = "base/trunk/src/meteo.F90"
GRIDLL = "proj/ct-unified/branches/speedup/src/grid_type_ll__speedup.F90"

# `diffusion_kernels` builds its constants out of `binas` and `dims` through `use`, so a
# name is resolved against this path in order, not against one file.
PHYS = (DIFFK, BINAS, DIMS)


# ----------------------------------------------------------------- reading the Fortran
@cache
def source(rel_path):
    path = TM5ZOOM / rel_path
    assert path.is_file(), f"{rel_path} is not in the checkout at {TM5ZOOM} — the mirror is broken"
    return path.read_text()


def _number(text):
    """A Fortran literal as a Python number; `1.0d-4` and `1.0e-4` are the same value."""
    text = text.strip().replace("d", "e").replace("D", "e")
    return float(text) if any(c in text for c in ".e") else int(text)


def _rhs(line, name):
    """The right-hand side of `<name> = ...` in a declaration, stopping at the next
    declaration on the same line (`betam=15.,betah=15.`) or at a comment."""
    tail = re.split(rf"(?<![\w]){name}\s*=", line, maxsplit=1, flags=re.IGNORECASE)[1]
    tail = tail.split("!")[0]
    depth, end = 0, len(tail)
    for i, ch in enumerate(tail):
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            end = i
            break
    return tail[:end].strip()


def declaration(name, files):
    """The RHS of the Fortran `parameter` declaration of `name`, and the file it came from."""
    pattern = re.compile(
        rf"^[ \t]*(?:integer|real|double\s+precision)\b[^!\n]*\bparameter\b[^!\n]*::[^!\n]*"
        rf"(?<![\w]){name}\s*=",
        re.IGNORECASE,
    )
    for rel_path in files:
        found = {_rhs(ln, name) for ln in source(rel_path).splitlines() if pattern.match(ln)}
        if found:
            assert len(found) == 1, f"{name} declared inconsistently in {rel_path}: {found}"
            return found.pop(), rel_path
    raise AssertionError(f"no `parameter :: {name} = ...` in {', '.join(files)} — mirror is broken")


IDENT = re.compile(r"(?<![\w.])[A-Za-z_]\w*")


def fortran_value(name, files, _seen=()):
    """The value of a Fortran parameter, resolving the names its RHS is built from.

    `fak = fr_excess * 8.5`, `rair = Rgas*1.e3/xmair`, `onet = 1./3.` — the declarations are
    expressions over other parameters, sometimes from another module.  Evaluating them in
    source order is what makes the comparison meaningful: if `Rgas` moves, `rair` moves here
    and does not move in the Python, and the test says so.
    """
    assert name.lower() not in _seen, f"{name}: circular Fortran declaration"
    rhs, _ = declaration(name, files)
    expr = rhs
    for ident in {m.group() for m in IDENT.finditer(rhs)}:
        value = fortran_value(ident, files, (*_seen, name.lower()))
        expr = re.sub(rf"(?<![\w.]){ident}(?![\w])", f"({value!r})", expr)
    expr = re.sub(r"(?<=[\d.])[dD](?=[-+\d])", "e", expr)
    try:
        return eval(compile(ast.parse(expr, mode="eval"), "<fortran>", "eval"), {"__builtins__": {}})
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - a new declaration form
        raise AssertionError(f"{name}: cannot evaluate Fortran RHS {rhs!r} -> {expr!r}") from exc


def assigned(rel_path, name):
    """Every distinct value assigned to a local, in order of first appearance."""
    found = re.findall(rf"^[ \t]*{name}\s*=\s*([\w.+-]+)", source(rel_path), re.IGNORECASE | re.M)
    assert found, f"no `{name} = ...` assignment in {rel_path} — the mirror is broken"
    out = []
    for v in (_number(x) for x in found):
        if v not in out:
            out.append(v)
    return tuple(out)


REAL = re.compile(r"(?<![\w.])\d[\d.]*(?:[eEdD][-+]?\d+)?")


def expression_literals(rel_path, anchor):
    """The real literals of the first statement matching `anchor`, in source order.

    Some constants are not named on either side: `krate_oh = 2.45e-12 * exp(-1775.0/T)`.
    Integer exponents (`cos(...)**2`) are not values of this kind and are left out.
    """
    for line in source(rel_path).splitlines():
        if re.search(anchor, line) and not line.lstrip().startswith("!"):
            nums = [t for t in REAL.findall(line.split("!")[0]) if "." in t or "e" in t.lower()]
            assert nums, f"{anchor} in {rel_path} has no real literals"
            return tuple(_number(t) for t in nums)
    raise AssertionError(f"no statement matching {anchor!r} in {rel_path} — the mirror is broken")


# ----------------------------------------------------------------- what mirrors what
# Keyed by the Python name, applied to every pydyn module that defines it: a constant that
# two modules disagree about is itself a bug, and this catches that too.
CONSTANTS = {
    "AE": ("param", "ae", PHYS),
    "ALFA_CHARNOCK": ("param", "alfa_charnock", PHYS),
    "ALFA_GARRET": ("param", "alfa_garret", PHYS),
    "APZERO": ("param", "apzero", PHYS),
    "AVOG": ("param", "Avog", (BINAS,)),
    "BETAH": ("param", "betah", PHYS),
    "BINM": ("param", "binm", PHYS),
    "CCON": ("param", "ccon", PHYS),
    "CKMIN": ("param", "ckmin", PHYS),
    "CP_AIR": ("param", "cp_air", PHYS),
    "DEG2RAD": ("param", "gtor", (DIMS, BINAS)),
    "EPSILO": ("param", "epsilo", PHYS),
    "FAK": ("param", "fak", PHYS),
    "FAKN": ("param", "fakn", PHYS),
    "GRAV": ("param", "grav", (BINAS,)),
    "GTOR": ("param", "gtor", (DIMS, BINAS)),
    "KVHMIN": ("assign", "kvhmin", DIFFK),
    "LVAP": ("param", "lvap", (BINAS,)),
    "MAX_GLOBAL_ITERATION": ("param", "max_global_iteration", (ADVM,)),
    "MAX_NLOOP_X": ("param", "cfl_max_nloop_x", (ADVM,)),
    "MAX_NLOOP_Y": ("param", "cfl_max_nloop_y", (ADVM,)),
    "MAX_NLOOP_Z": ("param", "cfl_max_nloop_z", (ADVM,)),
    "MB_MAX_TRIES": ("param", "MB_MAX_TRIES", (METEO,)),
    "POISSON_DIRECT_MAX": ("param", "POISSON_DIRECT_MAX", (GRIDLL,)),
    "NU_AIR": ("param", "nu_air", PHYS),
    "ONE": ("param", "one", (DIMS,)),
    "ONET": ("param", "onet", PHYS),
    "N_OPERATORS": ("param", "n_operators", (DIMS,)),
    "CFL_N_OPERATORS": ("param", "n_operators", (ADVM,)),
    "PI": ("param", "pi", (BINAS,)),
    "RAIR": ("param", "rair", PHYS),
    "RGAS": ("param", "Rgas", (BINAS,)),
    "RHO_AIR": ("param", "rho_air", PHYS),
    "RICR": ("param", "ricr", PHYS),
    "SFFRAC": ("param", "sffrac", PHYS),
    "SP_REGION0": ("param", "p_global", (BINAS,)),
    "SSFRAC": ("param", "ssfrac", PHYS),
    "TINY": ("param", "tiny", PHYS),
    "VKARMAN": ("param", "vkarman", (BINAS,)),
    "XMAIR": ("param", "xmair", (BINAS,)),
    "XMCH4": ("param", "xmch4", (CHEMP,)),
    "ZACB": ("param", "zacb", PHYS),
    "ZCRDQ": ("param", "zcrdq", PHYS),
    "ZKAPPA": ("param", "zkappa", PHYS),
}

# The sub-cycle caps of the advection kernels are defaults of a Python signature, not
# module constants, so they are listed by function rather than by name ("module.function"
# outside kernels.py: the zoom child kernels and region 1's row loop carry their own copy).
KERNEL_CAPS = {
    "advect_x.max_nloop": ("advect_x", "max_nloop", ADVX, "max_nloop"),
    "advect_y.max_nloop": ("advect_y", "max_nloop", ADVY, "max_nloop"),
    "advect_z.iter_limit": ("advect_z", "iter_limit", ADVZ, "iter_limit"),
    "zoom._dynamu_region1.max_nloop": ("zoom._dynamu_region1", "max_nloop", ADVX, "max_nloop"),
    "zoom._dynamu_child.max_nloop": ("zoom._dynamu_child", "max_nloop", ADVX, "max_nloop"),
    "zoom._dynamv_child.max_nloop": ("zoom._dynamv_child", "max_nloop", ADVY, "max_nloop"),
    "zoom._dynamw_child.iter_limit": ("zoom._dynamw_child", "iter_limit", ADVZ, "iter_limit"),
    "zoom._advectm_x.max_nloop": ("zoom._advectm_x", "max_nloop", ADVM, "cfl_max_nloop_x"),
    "zoom._advectm_y.max_nloop": ("zoom._advectm_y", "max_nloop", ADVM, "cfl_max_nloop_y"),
    "zoom._advectm_z.iter_limit": ("zoom._advectm_z", "iter_limit", ADVM, "cfl_max_nloop_z"),
}

# Pairs and expression literals, which do not fit one name to one declaration.
TUPLES = {
    "balance.MB_TOL_PRE+TIGHT rms": (
        lambda b: (b.MB_TOL_PRE[0], b.MB_TOL_TIGHT[0]), "balance", ("assign", METEO, "tol_rms")),
    "balance.MB_TOL_PRE+TIGHT max-difference": (
        lambda b: (b.MB_TOL_PRE[1], b.MB_TOL_TIGHT[1]), "balance", ("assign", METEO, "tol_diff")),
    # tm5-zoom 9c4779cba moved it to tp_lawrence (ch4loss.tropopause : lawrence, the default;
    # its wmo alternative is not transcribed); the first match in source order is the literal
    "chem tropopause pressure": (
        lambda c: (c.TROP_P0, c.TROP_DP), "chem",
        ("expr", CHEM, r"\b(tropopause_pressure|tp_lawrence)\s*=")),
    "chem OH rate constant": (
        lambda c: (c.OH_A, c.OH_EA), "chem", ("expr", CHEM, r"krate_oh\s*=")),
    "chemfields Hossaini Cl unit factors": (
        lambda c: (c.PPB, c.PER_CM3), "chemfields", ("expr", READOH, r"Cl_mm\s*=\s*\(")),
    "chemfields interpolation-map span floor": (
        lambda c: (c.MAP_EPS,), "chemfields", ("expr", REGRID, r"if \(\(s1 \+ s2\) >")),
    "initial ppm factor": (
        lambda i: (i.PPM,), "initial", ("expr", SOURCES, r"tracer_init\(region\)%d3 \* ")),
    "initial IC mean guard": (
        lambda i: i.IC_MEAN_RANGE, "initial", ("expr", SOURCES, r"if \( fld_mean <")),
    "chem tropospheric Cl rate constant": (
        lambda c: (c.CL_A, c.CL_T0, c.CL_N, c.CL_EA), "chem", ("expr", CHEM, r"krate_cl_trop\s*=")),
}

# Numeric constants that are deliberately not mirrored, with the reason. Anything not here
# and not in CONSTANTS fails test_every_numeric_constant_is_accounted_for.
WAIVED = {
    "FFTW_FORWARD": "FFTW's own API constant, not TM5's",
    "RELEASE_AFTER": "replay's memory window (events whose arrays are dropped), not a model value",
    "READ_AHEAD": "replay's read-ahead window (events read by a thread), not a model value",
    "WINDOW": "forward's reader-process window (the archive's 3-hourly instants), not a model value",
    "FFTW_BACKWARD": "FFTW's own API constant, not TM5's",
    "FFTW_ESTIMATE": "FFTW's own API constant, not TM5's",
    "TROP_P0": "mirrored as a tuple, see TUPLES",
    "TROP_DP": "mirrored as a tuple, see TUPLES",
    "OH_A": "mirrored as a tuple, see TUPLES",
    "OH_EA": "mirrored as a tuple, see TUPLES",
    "CL_A": "mirrored as a tuple, see TUPLES",
    "CL_T0": "mirrored as a tuple, see TUPLES",
    "CL_N": "mirrored as a tuple, see TUPLES",
    "CL_EA": "mirrored as a tuple, see TUPLES",
    "MB_TOL_PRE": "mirrored as a tuple, see TUPLES",
    "MB_TOL_TIGHT": "mirrored as a tuple, see TUPLES",
    "ECHLEVS": "an integer array parameter, see test_echlevs_matches_dims_levels",
    "PPB": "mirrored as a tuple, see TUPLES",
    "PER_CM3": "mirrored as a tuple, see TUPLES",
    "MAP_EPS": "mirrored as a tuple, see TUPLES",
    "PPM": "mirrored as a tuple, see TUPLES",
    "IC_MEAN_RANGE": "mirrored as a tuple, see TUPLES",
    # pydyn.gpu: launch shapes, opcodes and the benchmark's inputs, none a model value
    "NT_MAX": "csrc/pd.h's PD_NT_MAX, see test_device_tracer_cap_matches_pd_h",
    "SEARCH_ROUNDS": "gpu.kernels' sub-cycle search: rounds queued before one host read",
    "LINES": "gpu.coop's launch shape (lines a group)",
    "CVF_WARPS": "gpu.convec_gen's launch shape (warps a block)",
    "COPY": "gpu.boxes' opcode", "MUL": "gpu.boxes' opcode", "DIV": "gpu.boxes' opcode",
    "SEQSUM": "gpu.boxes' opcode", "SET": "gpu.boxes' opcode", "ADV": "gpu.boxes' opcode",
    "DIVMUL": "gpu.boxes' opcode", "ZSUM": "gpu.boxes' opcode", "ZSUMMUL": "gpu.boxes' opcode",
    "SECONDS_A_YEAR": "gpu.bench's unit for sim-yr/h (365 days), not a model value",
    "JRED": "gpu.bench's input: glb3x2's reduced grid as the fixture's dump records it; the "
            "replays read it from each dump (pydyn.redgrid)",
    "CLUSTSIZE": "gpu.bench's input, with JRED",
}

# Fortran-side limits with no Python counterpart, so there is nothing to compare. Recorded
# here rather than left unsaid, because "not mirrored" and "forgotten" look identical.
NO_COUNTERPART = {
    "nredmax": "an array bound in redgridZoom.F90 (60 -> 120 at 1x1); pydyn.redgrid sizes "
               "itself from the dump's meta and has no such limit",
    "max_global_iteration (mp)": "tm5-mp carries its own value; this repo transcribes zoom",
}


def numeric_constants():
    """(module name, constant name, value) for every uppercase numeric in the package and its
    subpackages (`gpu.kernels`, `zoom.tree`, ...)."""
    out = []
    for info in pkgutil.walk_packages(pydyn.__path__, "pydyn."):
        mod = importlib.import_module(info.name)
        for key, value in vars(mod).items():
            if not key.isupper() or key.startswith("_") or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) or (
                isinstance(value, tuple) and value and all(isinstance(x, (int, float)) for x in value)
            ):
                out.append((info.name.removeprefix("pydyn."), key, value))
    return sorted(out)


ROWS = [(m, k, v) for m, k, v in numeric_constants() if k in CONSTANTS]


# ----------------------------------------------------------------- the tests
@needs_checkout
@pytest.mark.parametrize(("module", "name", "mine"), ROWS, ids=[f"{m}.{k}" for m, k, _ in ROWS])
def test_constant_matches_fortran(module, name, mine):
    kind, fortran_name, where = CONSTANTS[name]
    theirs = fortran_value(fortran_name, where) if kind == "param" else assigned(where, fortran_name)[0]
    assert mine == theirs, (
        f"pydyn.{module}.{name} is {mine!r}; Fortran `{fortran_name}` is {theirs!r}. "
        f"The model moved; decide what the transcription should say before changing it."
    )


@needs_checkout
@pytest.mark.parametrize("label", sorted(KERNEL_CAPS))
def test_kernel_sub_cycle_cap_matches_fortran(label):
    import inspect

    fn_name, arg, path, fortran_name = KERNEL_CAPS[label]
    module, _, fn = fn_name.rpartition(".")
    owner = importlib.import_module(f"pydyn.{module or 'kernels'}")
    mine = inspect.signature(getattr(owner, fn)).parameters[arg].default
    theirs = fortran_value(fortran_name, (path,))
    assert mine == theirs, f"{owner.__name__}.{fn}({arg}=) is {mine!r}; {path} `{fortran_name}` is {theirs!r}"


@needs_checkout
def test_zoom_split_order_matches_dims():
    """zoom.SPLITORDER is dims.F90's `splitorder`, and zoom_mode 1 -- the only expansion
    define_splitorderzoom's transcription does -- is still the model's."""
    from pydyn import zoom

    decl = re.search(r"splitorder\s*=\s*&?\s*\(/(.*?)/\)", source(DIMS), re.S)
    assert decl, f"no splitorder declaration in {DIMS}"
    assert "".join(re.findall(r"'(\w)'", decl.group(1))) == zoom.SPLITORDER
    assert fortran_value("zoom_mode", (DIMS,)) == 1


@needs_checkout
def test_cfl_split_order_matches_advectm_cfl():
    """zoom.CFL_SPLITORDER is advectm_cfl.F90's own `splitorder`, the rehearsal's xyzzyx."""
    from pydyn import zoom

    decl = re.search(r"splitorder\(nsplitsteps\)\s*=\s*\(/(.*?)/\)", source(ADVM), re.S)
    assert decl, f"no splitorder declaration in {ADVM}"
    assert "".join(re.findall(r"'(\w)'", decl.group(1))) == zoom.CFL_SPLITORDER


@needs_checkout
def test_echlevs_matches_dims_levels():
    from pydyn import chemfields

    decl = re.search(r"echlevs\(0:\d+\)\s*=\s*\(/(.*?)/\)", source(LEVELS), re.S)
    assert decl, f"no echlevs declaration in {LEVELS}"
    theirs = tuple(int(x) for x in re.findall(r"\d+", re.sub(r"&", " ", decl.group(1))))
    assert chemfields.ECHLEVS == theirs


@needs_checkout
@pytest.mark.parametrize("label", sorted(TUPLES))
def test_tuple_matches_fortran(label):
    getter, module_name, spec = TUPLES[label]
    mine = getter(importlib.import_module(f"pydyn.{module_name}"))
    theirs = assigned(spec[1], spec[2]) if spec[0] == "assign" else expression_literals(spec[1], spec[2])
    assert mine == theirs, f"{label}: pydyn has {mine!r}, the Fortran has {theirs!r}"


def test_every_numeric_constant_is_accounted_for():
    """A constant added to pydyn without a row here would be unmirrored and unnoticed."""
    loose = {
        f"{module}.{name}"
        for module, name, _ in numeric_constants()
        if name not in CONSTANTS and name not in WAIVED
    }
    assert not loose, (
        f"not mirrored and not waived: {sorted(loose)}. Add a CONSTANTS row naming the "
        f"Fortran declaration it copies, or a WAIVED entry saying why it has no source."
    )


@needs_checkout
def test_every_mirror_is_reachable():
    """A source file that moved makes a mirror unverifiable, which is a failure, not a skip."""
    files = {f for _, _, where in CONSTANTS.values() for f in ((where,) if isinstance(where, str) else where)}
    files |= {path for _, _, path, _ in KERNEL_CAPS.values()}
    files |= {spec[1] for _, _, spec in TUPLES.values()}
    for rel_path in sorted(files):
        assert (TM5ZOOM / rel_path).is_file(), f"{rel_path} is gone from the checkout"
    assert NO_COUNTERPART, "keep the record of Fortran limits that have no Python side"


def test_device_tracer_cap_matches_pd_h():
    """gpu.kernels.NT_MAX, which refuses a tracer count, is the kernels' own PD_NT_MAX."""
    from pydyn.gpu import kernels

    header = (pydyn.__path__[0] + "/gpu/csrc/pd.h")
    found = re.search(r"#define\s+PD_NT_MAX\s+(\d+)", open(header).read())
    assert found and int(found.group(1)) == kernels.NT_MAX
