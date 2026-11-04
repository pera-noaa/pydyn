"""Run one production Fortran kernel through fortran/harness and compare.

    write_case(dir, "x", state, am=am)
    out = run(dir)                        # dict of pydyn-layout arrays + stats
    report = compare(out["rm"], rm_py)    # bitwise report with ulp distance

The Fortran side reads the halo-layout stream files written here (see layout.py), calls
the kernel through the same entry points as the model (advectxzoom/advectyzoom/
advectzzoom/dynam0/convec for region 1) and writes the modified arrays back.

The zoom harness (docs/state.md §3d) is driven the same way by write_zoom_case / run_zoom:
every region's arrays with their Fortran bounds (zoom.ZoomState), one call -- an exchange
routine, an advection of any region, Setup_MassFlow, Proces_Region, a step or Check_CFL --
through fortran/harness_zoom<N>, N the number of regions.
"""

import re
import subprocess
from pathlib import Path

import numpy as np

from .layout import from_tm5, read_fortran_stream, tm5_shape, to_tm5, write_fortran_stream

FORTRAN_DIR = Path(__file__).resolve().parent.parent / "fortran"
BINARY = FORTRAN_DIR / "harness"

STATE = ("m", "rm", "rxm", "rym", "rzm")
FLUX = {"x": "am", "y": "bm", "z": "cm"}
CONV = ("entu", "detu", "entd", "detd", "dkg")
CONV_INT = ("cloud_top", "cloud_lfs", "cloud_base")


class HarnessError(RuntimeError):
    """The Fortran binary exited with an error (escape_tm, goExit, ...); .stderr has the text."""

    def __init__(self, returncode, stderr):
        super().__init__(f"harness exited {returncode}: {stderr.strip()}")
        self.returncode = returncode
        self.stderr = stderr


def available():
    return BINARY.exists()


def write_case(case_dir, kernel, state, *, ndyn=3600, limits=True, lmax_conv=None, debug=False, **inputs):
    """Write `<case_dir>/case.nml` and the input stream files for one kernel call.

    kernel: 'x' | 'y' | 'z' | 'dynam0' | 'convec'.  state: pydyn State (interior layout).
    inputs, in pydyn layout: am / bm / cm for the advection kernels; pu, pv, bt for dynam0;
    conv=dict(entu, detu, entd, detd, dkg, cloud_top, cloud_lfs[, cloud_base]) for convec.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    im, jm, lm = state.m.shape
    nt = state.rm.shape[3]
    lmax_conv = lm if lmax_conv is None else lmax_conv

    def write(name, arr, dtype="<f8"):
        write_fortran_stream(case_dir / f"{name}.bin", arr, dtype)

    for name in STATE:
        write(name, to_tm5(name, getattr(state, name), im, jm, lm))
    if kernel in FLUX:
        name = FLUX[kernel]
        write(name, to_tm5(name, inputs[name], im, jm, lm))
    elif kernel == "dynam0":
        for name in ("pu", "pv", "bt"):
            write(name, np.asarray(inputs[name], dtype=float))
    elif kernel == "convec":
        conv = dict(inputs["conv"])
        conv.setdefault("cloud_base", np.zeros((im, jm), np.int32))
        for name in CONV:
            write(name, conv[name])
        for name in CONV_INT:
            write(name, conv[name], "<i4")
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    (case_dir / "case.nml").write_text(
        "&case\n"
        f" kernel = '{kernel}'\n"
        f" im_ = {im}, jm_ = {jm}, lm_ = {lm}, nt = {nt}\n"
        f" ndyn_ = {int(ndyn)}, lmax_conv = {lmax_conv}\n"
        f" limits_ = {'.true.' if limits else '.false.'}, debug = {'.true.' if debug else '.false.'}\n"
        "/\n"
    )
    return case_dir


def run(case_dir, binary=BINARY):
    """Run the harness on a written case; return the outputs in pydyn layout.

    Keys: m, rm, rxm, rym, rzm; am/bm/cm as applicable; 'stats' = dict(nloop_max, xi, nxi)
    each a 3-vector (x, y, z) from dims; 'stdout'.  Raises HarnessError on nonzero exit.
    """
    case_dir = Path(case_dir)
    proc = subprocess.run([str(binary), str(case_dir)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise HarnessError(proc.returncode, proc.stderr + proc.stdout)

    nml = dict(re.findall(r"(\w+?)_?\s*=\s*(\d+)", (case_dir / "case.nml").read_text()))
    im, jm, lm, nt = (int(nml[k]) for k in ("im", "jm", "lm", "nt"))

    def read(name):
        full = read_fortran_stream(case_dir / f"out_{name}.bin", tm5_shape(name, im, jm, lm, nt))
        return from_tm5(name, full, im, jm, lm)

    out = {name: read(name) for name in STATE}
    out.update({name: read(name) for name in FLUX.values() if (case_dir / f"out_{name}.bin").exists()})
    stats = (line.split() for line in (case_dir / "out_stats.txt").read_text().splitlines())
    out["stats"] = {key: np.array(vals, dtype=float) for key, *vals in stats}
    out["stdout"] = proc.stdout
    return out


def _ulp_distance(a, b):
    """Number of representable doubles between a and b, elementwise (0 where equal)."""

    def ordered(x):  # sign-magnitude integer image -> monotone ordering
        i = x.astype("<f8").view(np.int64)
        return np.where(i < 0, np.iinfo(np.int64).min - i, i)

    return np.abs(ordered(a) - ordered(b))


def compare(fortran, python, name=""):
    """Bitwise comparison report: dict(equal, n_diff, max_ulp, max_abs, max_rel, first, first_values)."""
    fortran = np.asarray(fortran, dtype=float)
    python = np.asarray(python, dtype=float)
    if fortran.shape != python.shape:
        raise ValueError(f"{name}: shape {fortran.shape} vs {python.shape}")
    differ = ~((fortran == python) | (np.isnan(fortran) & np.isnan(python)))
    if not differ.any():
        return {"name": name, "equal": True, "n_diff": 0, "max_ulp": 0, "max_abs": 0.0, "max_rel": 0.0}
    diff = np.abs(fortran - python)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(differ, diff / np.maximum(np.abs(fortran), np.abs(python)), 0.0)
    first = tuple(int(i) for i in np.argwhere(differ)[0])
    return {
        "name": name,
        "equal": False,
        "n_diff": int(differ.sum()),
        "max_ulp": int(_ulp_distance(fortran, python).max()),
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "first": first,
        "first_values": (float(fortran[first]), float(python[first])),
    }


def format_report(reports):
    def line(r):
        if r["equal"]:
            return f"{r['name']:>4}: bitwise equal"
        f, p = r["first_values"]
        return (
            f"{r['name']:>4}: {r['n_diff']} differ, max {r['max_ulp']} ulp, max rel {r['max_rel']:.3e}, "
            f"first at {r['first']} fortran={f!r} python={p!r}"
        )

    return "\n".join(line(r) for r in reports)


# ------------------------------------------------------------------ the zoom harness (§3d)
def zoom_binary(nregions):
    """fortran/harness_zoom<N>: one binary per region count, as production's nregions is the
    true count (Check_CFL and Setup_MassFlow loop over all of them)."""
    return FORTRAN_DIR / f"harness_zoom{nregions}"


ZOOM_OUT = ("m", "rm", "rxm", "rym", "rzm", "am", "bm", "cm", "sp", "phlb")
ZOOM_IN = (*ZOOM_OUT, "pu", "pv", "dxyp")


def zoom_available(nregions=(2, 3)):
    return all(zoom_binary(n).exists() for n in nregions)


def write_zoom_case(
    case_dir, kernel, zoom, st, target, *, ndyn=3600, limits=True, conv=None, debug=False,
    n_steps=1, nread=3600,
):
    """Write a multi-region case for fortran/harness_zoom: every region's arrays with their
    Fortran bounds (zoom.ZoomState), the region tree of `zoom` (a zoom.Zoom), and the call.

    kernel: 'x' | 'y' | 'z' | 'put_xedges' | 'put_yedges' | 'mix_edges' | 'update_parent' |
    'coarsen_region' | 'coarsen_ambm' | 'm2phlb' | 'm2phlb1' | 'dynam0' | 'convec' |
    'proces_region' | 'step' | 'check_cfl' (n_steps steps of ndyn in nread; run_zoom returns
    the result as fields["cfl"] = (ndyn, n)), called for region `target` with the module state
    zoom.status / zoom.splitorderzoom.  conv = dict(entu, detu, entd, detd, dkg, cloud_top, cloud_lfs
    [, cloud_base], lmax) in the kernels.convec layout, of region `target`, or {region: dict}
    for several (the driver kernels convect every region).
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    def write(name, arr, dtype="<f8"):
        write_fortran_stream(case_dir / f"{name}.bin", arr, dtype)

    for r in zoom.regions():
        for name in ZOOM_IN:
            write(f"{name}_r{r}", st[r][name].data)
        write(f"zoomed_r{r}", zoom.zoomed[r], "<i4")
        write(f"edge_r{r}", zoom.edge[r], "<i4")
    write("at", zoom.at)
    write("bt", zoom.bt)
    lmax = zoom.lm
    if conv is not None:
        per_region = conv if all(isinstance(k, int) for k in conv) else {target: conv}
        for r, cv in per_region.items():
            cv = dict(cv)
            lmax = cv["lmax"]
            g = zoom.g(r)
            cv.setdefault("cloud_base", np.zeros((g.im, g.jm), np.int32))
            for name in CONV:
                write(f"{name}_r{r}", cv[name])
            for name in CONV_INT:
                write(f"{name}_r{r}", cv[name], "<i4")

    def ints(values):
        return ", ".join(str(int(v)) for v in values)

    regs = list(zoom.regions())
    g = [zoom.g(r) for r in regs]
    per = {
        "im_": [x.im for x in g], "jm_": [x.jm for x in g], "parent_": [x.parent for x in g],
        "xcyc_": [x.xcyc for x in g],
        "touch_sp_": [x.touch_sp for x in g], "touch_np_": [x.touch_np for x in g],
        "xref_": [x.xref for x in g], "yref_": [x.yref for x in g], "zref_": [x.zref for x in g],
        "tref_": [x.tref for x in g],
        "ibeg_": [zoom.ibeg.get(r, 1) for r in regs], "iend_": [zoom.iend.get(r, 1) for r in regs],
        "jbeg_": [zoom.jbeg.get(r, 1) for r in regs], "jend_": [zoom.jend.get(r, 1) for r in regs],
        "lbeg_": [zoom.lbeg.get(r, 1) for r in regs], "lend_": [zoom.lend.get(r, zoom.lm) for r in regs],
        "isr_": [zoom.isr[r] for r in regs], "ier_": [zoom.ier[r] for r in regs],
        "jsr_": [zoom.jsr[r] for r in regs], "jer_": [zoom.jer[r] for r in regs],
        "status_": [zoom.status[r] for r in regs],
    }
    lines = ["&zoom", f" kernel = '{kernel}', target = {target}, nreg = {zoom.nregions}, nt = {st.nt}",
             f" lm_ = {zoom.lm}, ndyn_ = {int(ndyn)}, lmax_conv = {lmax}",
             f" n_steps = {int(n_steps)}, nread_ = {int(nread)}",
             f" limits_ = {'.true.' if limits else '.false.'}, debug = {'.true.' if debug else '.false.'}"]
    lines += [f" {k} = {ints(v)}" for k, v in per.items()]
    lines += [f" splitorder_({r}) = '{zoom.splitorderzoom[r]}'" for r in regs]
    for r in regs:
        kids = zoom.children[r]
        lines.append(f" children_({r},0) = {len(kids)}")
        lines += [f" children_({r},{k}) = {c}" for k, c in enumerate(kids, start=1)]
    lines.append("/")
    (case_dir / "case.nml").write_text("\n".join(lines) + "\n")
    return case_dir


def run_zoom(case_dir, zoom, st, binary=None):
    """Run harness_zoom on a written case; return (fields, stats): fields[r][name] as FViews
    shaped like st[r][name], stats[r] = dict(nloop_max, xi, nxi), each an (x, y, z) 3-vector."""
    from .zoom import FView

    case_dir = Path(case_dir)
    binary = zoom_binary(zoom.nregions) if binary is None else binary
    proc = subprocess.run([str(binary), str(case_dir)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise HarnessError(proc.returncode, proc.stderr + proc.stdout)
    fields = {}
    for r in zoom.regions():
        fields[r] = {}
        for name in ZOOM_OUT:
            ref = st[r][name]
            data = read_fortran_stream(case_dir / f"out_{name}_r{r}.bin", ref.data.shape)
            fields[r][name] = FView(data, ref.lower)
    cfl = case_dir / "out_cfl.txt"
    if cfl.exists():  # 'check_cfl': the (ndyn, n) Check_CFL settled on
        vals = dict(line.split() for line in cfl.read_text().splitlines())
        fields["cfl"] = (int(vals["ndyn"]), int(vals["n"]))
    stats = {r: {} for r in zoom.regions()}
    for line in (case_dir / "out_stats.txt").read_text().splitlines():
        key, *vals = line.split()
        name, r = re.fullmatch(r"([a-z_]+?)(\d+)", key).groups()
        stats[int(r)][name] = np.array(vals, dtype=float)
    return fields, stats
