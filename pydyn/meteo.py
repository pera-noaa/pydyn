"""The meteo ingestion: from the tm5-nc archive on disk to the arrays the dynamic core
consumes (meteo.F90 Meteo_Setup_Mass / Meteo_Setup_Other_Core / Meteo_Produce_*, the
tmm/meteodata layers underneath, for a run whose archive is already on the model grid and
levels, as the production `ch4_6x4_fwd.rc` is: no horizontal or vertical regridding).

Everything reduces to six rules:

    records      one file per day and parameter (SOURCES: `<param>_<yyyymmdd>_00p03.nc` for
                 the model-level fields, 8 records at 0,3,...,21 h; `_00p01.nc` for the 1x1
                 surface fields, 24 hourly records; one file per month for srols, one
                 timeless file for oro and lsm); a record is picked by its (t1, t2) time
                 bounds exactly (GetTimeRecordNumber): instantaneous fields have bounds
                 (t, t), the accumulated ones the interval they average.  float32 in the
                 file, exact once widened to real8; a unit factor only where the file's unit
                 is not the model's (FACTORS: lsm 1 -> %).
    interpN      a read interval tr = (t1, t2) uses the instants tprev = floor(t1) and
                 tnext = ceil(t2) to whole N h (SetupSetup); the field is
                     alfa1 data(tprev) + alfa2 data(tnext)        (TimeInterpolation)
                 with InterpolFractions at tmid = t1 + (t2 - t1)/2.  When tprev == tnext
                 the fractions are 1/2, 1/2 of the same record.  N = 3 for the model
                 levels, 1 for the surface.
    averN        the field of the N-h interval that contains t1 (tc1 = floor(t1)); the
                 hourly read interval never crosses it, so no weights are involved.
    area-aver    a 1x1 surface field goes to the 6x4 grid as the area-weighted mean of the
                 24 cells under each model cell, the weights the 1x1 rows' area_m2, the
                 sums in the file's index order (FillGrid 'area-aver'), after the unit
                 factor and before the time interpolation.
    sp match     the surface pressure of one instant is shifted by a constant so that its
                 area-weighted global mean is `sp_region0` = 98500 Pa (Match 'area-aver'
                 against the one-cell global grid), the sums in index order.
    products     phlb(l) = a(l) + b(l) sp, m = (phlb(l) - phlb(l+1)) / g area_m2 at the first
                 read only (Pressure_to_Mass); the convection fluxes are turned into per-cell
                 kg/s with `dxyp` and closed column by column at their largest level
                 (ConvUnitConvert), and the cloud levels are read off detu / entd
                 (ConvCloudDim); dkg from the surface fields, T, Q, the step's pu / pv and
                 the surface pressure at the middle of the step (Calc_Kzz, diffusion.py).

Layout is pydyn's (layout.py): mfu (im+1, jm, lm) = mfu(0:im), mfv (im, jm+1, lm),
mfw (im, jm, lm+1) = mfw(0:lm), sp (im, jm), temper (im, jm, lm), the convection
parameters (im, jm, lmax_conv), cloud levels (im, jm) 1-based with 0 = none, the surface
fields (im, jm).

The initial tracer field (istart 66) is not meteo: initial.py.
"""

import shutil
import weakref
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from . import diffusion
from .balance import GRAV, PI, interpol_fractions, time_interpolation
from .grid import AE, GLB6X4, ONE_BY_ONE, fill, globe_area
from .ops import libm, seqsum

__all__ = [
    "SP_REGION0",
    "Archive",
    "Meteo",
    "Stager",
    "cloud_dims",
    "conv_unit_convert",
    "dxyp",
    "hybrid_coefficients",
    "instants",
    "interval",
    "match_global_mean",
    "month",
    "pressure_to_mass",
]

SP_REGION0 = 98500.0  # binas.F90 p_global: the global-mean sp the archive's sp is matched to

# The archive's files per parameter (the rc's meteo.class / sourcekey), `{p}` the parameter,
# `{t}` the time of the record.
ML = "ec/ea/h06h18tr3/tropo34/glb600x400/{t:%Y}/{t:%m}/{p}_{t:%Y%m%d}_00p03.nc"
SFC = "ec/ea/h06h18tr1/sfc/glb100x100/{t:%Y}/{t:%m}/{p}_{t:%Y%m%d}_00p01.nc"
AN = "ec/ea/an0tr1/sfc/glb100x100/{t:%Y}/{t:%m}/{p}_{t:%Y%m%d}_00p01.nc"
MONTHLY = "ec/ea/an0tr1/sfc/glb100x100/{t:%Y}/{t:%m}/{p}_{t:%Y%m}.nc"
CONST = "ec/ea/an0tr1/sfc/glb100x100/{p}.nc"
SOURCES = {
    "mfuv": ML, "mfw": ML, "sp": ML, "t": ML, "q": ML, "convec": ML,
    "u10m": SFC, "v10m": SFC, "sshf": SFC, "slhf": SFC, "ewss": SFC, "nsss": SFC,
    "sr": AN, "srols": MONTHLY, "oro": CONST, "lsm": CONST,
}  # fmt: skip
# The archive's variables per parameter file (the parameter's own name when absent) and their
# pydyn layout, (lon, lat[, lev]).
VARIABLES = {
    "mfuv": ("mfu", "mfv"),
    "mfw": ("mfw",),
    "sp": ("sp",),
    "t": ("T",),
    "q": ("Q",),
    "convec": ("eu", "du", "ed", "dd"),
}
# The rc's meteo.tinterp.* per parameter.
TINTERP = {
    "sp": "interp3", "mfuv": "interp3", "mfw": "interp3", "t": "interp3", "q": "interp3", "convec": "aver3",
    "u10m": "interp1", "v10m": "interp1", "sr": "interp1",
    "sshf": "aver1", "slhf": "aver1", "ewss": "aver1", "nsss": "aver1",
    "srols": "month", "oro": "const", "lsm": "const",
}  # fmt: skip
# udunits conversion from the file's unit to the model's, where they differ (tmm_mf_tm5_nc).
FACTORS = {"lsm": 100.0}  # 1 -> %
SURFACE = ("oro", "lsm", "sr", "srols", "u10m", "v10m", "sshf", "slhf", "ewss", "nsss")  # Calc_Kzz's


# ------------------------------------------------------------------ time rules (SetupSetup)
def _floor(t, dth):
    """t truncated to the previous multiple of dth hours (baseh = 0)."""
    return t.replace(hour=int((t.hour + t.minute / 60.0) // dth) * dth, minute=0, second=0)


def instants(tr, dth=3):
    """'interpN': the archive instants (tprev, tnext) around the read interval tr = (t1, t2):
    t1 rounded down and t2 rounded up to whole dth hours."""
    t1, t2 = tr
    tprev, tnext = _floor(t1, dth), _floor(t2, dth)
    if tnext < t2:
        tnext += timedelta(hours=dth)
    return tprev, tnext


def interval(t, dth=3):
    """'averN': the dth-hour archive interval (tc1, tc2) that contains t."""
    tc1 = _floor(t, dth)
    return tc1, tc1 + timedelta(hours=dth)


def month(t):
    """'month': the calendar month (t1, t2) that contains t."""
    t1 = t.replace(day=1, hour=0, minute=0, second=0)
    return t1, (t1 + timedelta(days=32)).replace(day=1)


def interpolate(tr, tprev, data1, tnext, data2):
    """TimeInterpolation 'interp*': the field at the middle of tr from the records at
    tprev and tnext."""
    t1, t2 = tr
    alfa1, alfa2 = interpol_fractions(t1 + (t2 - t1) / 2, tprev, tnext)
    return time_interpolation(data1, data2, alfa1, alfa2)


# ------------------------------------------------------------------ the archive
class Archive:
    """The tm5-nc meteo archive under `root`, one file per parameter and day / month / ever
    (SOURCES), records found by their `timevalues_bounds`."""

    KEEP = 3  # records kept per parameter: consecutive read intervals share a 3-hourly instant
    OPEN = 64  # files kept open, the oldest closed past it (a day opens ~16)

    def __init__(self, root, sources=SOURCES, stage=None):
        self.root, self.sources = Path(root), sources
        self.stage = None if stage is None else Path(stage)  # a Stager's copies, read if there
        self._open = {}
        self._records = {}  # a file's {(t1, t2): record index}, from its timevalues_bounds once
        self._kept = {}  # param -> {(t1, t2): its variables, read-only}, the last KEEP read

    def path(self, param, t=None):
        return self.root / self.sources[param].format(p=param, t=t)

    def file(self, param, t=None):
        path = self.path(param, t)
        nc = self._open.get(path)
        if nc is None:
            src = path
            if self.stage is not None:
                staged = self.stage / path.relative_to(self.root)
                if staged.exists():
                    src = staged
            if len(self._open) >= self.OPEN:
                self._open.pop(next(iter(self._open))).close()
            nc = self._open[path] = Dataset(src)
        return nc

    def record(self, param, t1, t2):
        """GetTimeRecordNumber: (file, record index) of the record with bounds (t1, t2)."""
        nc = self.file(param, t1)
        path = self.path(param, t1)
        if path not in self._records:
            bounds = nc["timevalues_bounds"][:].filled()
            index = {}
            for irec, b in enumerate(bounds):
                index.setdefault((datetime(*b[:, 0]), datetime(*b[:, 1])), irec)  # the first, as the scan
            self._records[path] = index
        irec = self._records[path].get((t1, t2))
        if irec is None:
            raise LookupError(f"{path}: no record with bounds {t1} - {t2}")
        return nc, irec

    def read(self, param, t1, t2):
        """The record's variables as float64 (lon, lat[, lev]) arrays in the model's units:
        {var: array}, read-only (kept for the next reads of the record)."""
        kept = self._kept.setdefault(param, {})
        out = kept.get((t1, t2))
        if out is None:
            nc, irec = self.record(param, t1, t2)
            out = {v: self._widen(nc[v][irec], param) for v in VARIABLES.get(param, (param,))}
            for a in out.values():
                a.flags.writeable = False
            if len(kept) >= self.KEEP:
                kept.pop(next(iter(kept)))
            kept[(t1, t2)] = out
        return dict(out)

    def read_instant(self, param, t):
        return self.read(param, t, t)

    def read_const(self, param):
        """A timeless field (oro, lsm), (lon, lat) in the model's units."""
        return self._widen(self.file(param)[param][:], param)

    @staticmethod
    def _widen(var, param):
        data = np.ascontiguousarray(var.filled().astype(np.float64).T)
        return data * FACTORS[param] if param in FACTORS else data

    def hybrid(self, param, t):
        """(a, b) of the lm+1 half levels, surface first (levi%a, levi%b), from the file's
        `ap_bounds`, `b_bounds`; bitwise the model's table for the production archive."""
        nc = self.file(param, t)
        return hybrid_coefficients(nc["ap_bounds"][:].filled(), nc["b_bounds"][:].filled())


class Stager:
    """Copies of the archive's files a run reads next on fast local storage, made by threads
    ahead of the readers: on a parallel file system a whole file's sequential copy costs far
    less than netCDF's many small reads.  The copies go in a new directory under `under`
    (e.g. a node's $TMPDIR or /dev/shm), `self.stage`, which an Archive given it reads from
    once a copy is complete (written aside, then renamed), the original until then.  ahead(t)
    each read interval: the days from t's to `days` after it, and the month's files; copies
    of days before yesterday are removed, and close() removes the directory."""

    def __init__(self, root, under, days=2, threads=8, sources=SOURCES):  # /work2: 617 MB/s at 8
        import tempfile
        from concurrent.futures import ThreadPoolExecutor

        self.root, self.days, self.sources = Path(root).resolve(), days, sources
        Path(under).mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix="pydyn-stage-", dir=under))
        self._removal = weakref.finalize(self, shutil.rmtree, self.stage, ignore_errors=True)  # at exit too
        self.pool = ThreadPoolExecutor(threads)
        self.staged = {}  # day -> the relative paths of its copies
        self.copies = []  # the copies under way
        self.months = set()  # months whose files (monthly, timeless) are staged
        self.failed = None  # the first copy's error: staging stops, the originals are read

    def _copy(self, rel):
        src, dst = self.root / rel, self.stage / rel
        if self.failed is not None or dst.exists() or not src.exists():
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, part)
            part.replace(dst)
        except OSError as e:  # a full stage: read the originals
            self.failed = self.failed or e

    def _stage(self, t, daily):
        rels = sorted({tpl.format(p=p, t=t) for p, tpl in self.sources.items() if ("%d}" in tpl) == daily})
        self.copies = [c for c in self.copies if not c.done()]
        self.copies += [self.pool.submit(self._copy, rel) for rel in rels]
        return rels

    def wait(self):
        """Until the copies asked for so far are made (or failed)."""
        for c in self.copies:
            c.result()

    def ahead(self, t):
        if (t.year, t.month) not in self.months:
            self.months.add((t.year, t.month))
            self._stage(t, daily=False)
        today = t.date()
        for k in range(self.days + 1):
            day = today + timedelta(days=k)
            if day not in self.staged:
                self.staged[day] = self._stage(datetime.combine(day, datetime.min.time()), daily=True)
        for day in [d for d in self.staged if d < today - timedelta(days=1)]:
            for rel in self.staged.pop(day):
                (self.stage / rel).unlink(missing_ok=True)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self._removal()


def hybrid_coefficients(ap_bounds, b_bounds):
    """levi%a, levi%b (0:lm) from the file's per-layer (lm, 2) bounds."""
    return (
        np.concatenate([ap_bounds[:1, 0], ap_bounds[:, 1]]),
        np.concatenate([b_bounds[:1, 0], b_bounds[:, 1]]),
    )


# ------------------------------------------------------------------ geometry
def dxyp(jm, dx, dy, ybeg):
    """geometry.F90: region_dat%dxyp (jm,) in m^2, accumulated latitude by latitude."""
    gtor = PI / 180.0
    dxx, dyy = dx * gtor, dy * gtor
    lat = np.empty(jm + 1)
    lat[0] = ybeg * gtor
    for j in range(jm):
        lat[j + 1] = lat[j] + dyy
    return dxx * (libm.sin(lat[1:]) - libm.sin(lat[:-1])) * AE**2


# ------------------------------------------------------------------ surface pressure and mass
def match_global_mean(sp, area, area_globe, sp_global=SP_REGION0):
    """Match 'area-aver' of sp (im, jm) against the one-cell global grid: sp shifted by the
    constant that makes its area-weighted mean `sp_global`.  The row sums and their weighted
    sum are accumulated in index order."""
    fsum = seqsum(seqsum(sp, axis=0) * area)
    fsum = fsum / area_globe
    return sp + (sp_global - fsum)


def pressure_to_mass(sp, a, b, area_m2):
    """Pressure_to_Mass: the half-level pressures phlb (im, jm, lm+1) and the air mass
    m (im, jm, lm) in kg of the columns under sp (im, jm)."""
    phlb = a + b * sp[..., None]
    m = phlb[..., :-1] - phlb[..., 1:]
    m = m / GRAV
    return phlb, m * area_m2[None, :, None]


# ------------------------------------------------------------------ convection
def conv_unit_convert(entu, detu, entd, detd, dxyp, czeta=1.0):
    """Meteo_Produce_ConvUnitConvert: the entrainment / detrainment rates (im, jm, lmax) from
    kg/m2/s to kg/s per cell, each column closed (sum of entrainment == sum of detrainment)
    by adjusting the level of its largest entrainment -- the first such level for the updraft,
    the last level (lmax) when the downdraft has none."""
    entu, detu, entd, detd = ((q * dxyp[None, :, None]) * czeta for q in (entu, detu, entd, detd))
    _close(entu, detu, np.argmax(entu, axis=-1))
    lmax = entd.shape[-1]
    lsave = np.where(np.any(entd > 0.0, axis=-1), np.argmax(entd, axis=-1), lmax - 1)
    _close(entd, detd, lsave)
    return entu, detu, entd, detd


def _close(ent, det, lsave):
    """ent(lsave) -= tote - totd, columnwise, in place."""
    tote, totd = seqsum(ent, axis=-1), seqsum(det, axis=-1)
    at = (*np.indices(lsave.shape), lsave)
    ent[at] = ent[at] - tote + totd


def cloud_dims(detu, entd):
    """Meteo_Produce_ConvCloudDim: (cloud_top, cloud_lfs, cloud_base) as 1-based levels, 0 when
    the column has no updraft detrainment / downdraft entrainment: the top is the highest
    level with detu > 0, the base the lowest, the level of free sinking the highest with
    entd > 0."""
    lmax = detu.shape[-1]

    def highest(q):
        return np.where(np.any(q > 0.0, axis=-1), lmax - np.argmax((q > 0.0)[..., ::-1], axis=-1), 0)

    base = np.where(np.any(detu > 0.0, axis=-1), np.argmax(detu > 0.0, axis=-1) + 1, 0)
    return highest(detu), highest(entd), base


# ------------------------------------------------------------------ the driver
class Meteo:
    """The meteo products of one region and read interval, from an Archive: what
    Meteo_Setup_Mass and Meteo_Setup_Other_Core leave for the dynamic core.

    `mass(tr)` returns the raw (unbalanced) mass fluxes at the two archive instants around
    the read interval and the matched surface pressures at its ends -- the inputs of
    balance.setup_mass -- plus, on the first call, (sp, phlb, m) of the initial air mass."""

    def __init__(self, archive, ll=GLB6X4, lmax_conv=26, czeta=1.0):
        self.archive, self.ll, self.lmax_conv, self.czeta = archive, ll, lmax_conv, czeta
        self.area = ll.area  # lli%area, rad^2
        self.area_globe = globe_area()
        self.area_m2 = ll.area_m2  # lli%area_m2
        self.dxyp = dxyp(ll.jm, ll.dlon, ll.dlat, ll.south)  # region_dat%dxyp
        self.first = True  # the next mass() is the run's first: it also sets up the air mass
        self._levels = None
        self._sp = {}  # instant -> matched surface pressure
        self._const = {}  # the timeless surface fields, once regridded
        self._surface = {}  # param -> {(t1, t2): field}, the last records (an hour's end, the next's start)

    def _interp3(self, param, var, tr):
        tprev, tnext = instants(tr)
        data = {t: self.archive.read_instant(param, t)[var] for t in (tprev, tnext)}
        return interpolate(tr, tprev, data[tprev], tnext, data[tnext])

    def levels(self, t):
        """(a, b) of the half levels."""
        if self._levels is None:
            self._levels = self.archive.hybrid("mfuv", t)
        return self._levels

    def surface_pressure(self, t):
        """sp at the instant t: interp3 of the archive's sp, matched to the global mean."""
        if t not in self._sp:
            self._sp[t] = match_global_mean(self._interp3("sp", "sp", (t, t)), self.area, self.area_globe)
        return self._sp[t]

    def mass(self, tr):
        """Meteo_Setup_Mass, before balancing: {mfu1, mfv1, mfw1, mfu2, mfv2, mfw2, sp1, sp2,
        instants}, and on the first call the initial sp, phlb, m."""
        out = {"instants": instants(tr)}
        for k, t in enumerate(out["instants"], 1):
            out.update({f"{v}{k}": q for v, q in self.archive.read_instant("mfuv", t).items()})
            out[f"mfw{k}"] = self.archive.read_instant("mfw", t)["mfw"]
        out["sp1"], out["sp2"] = (self.surface_pressure(t) for t in tr)
        if self.first:
            a, b = self.levels(tr[0])
            out["sp"] = out["sp1"]
            out["phlb"], out["m"] = pressure_to_mass(out["sp"], a, b, self.area_m2)
            self.first = False
        return out

    def spm(self, tr, tread=None):
        """Calc_Kzz's spm: the matched surface pressures sp1, sp2 at the ends of the read
        interval `tread` (tr itself if None), interpolated to the middle of the step tr."""
        t1, t2 = tr if tread is None else tread
        return interpolate(tr, t1, self.surface_pressure(t1), t2, self.surface_pressure(t2))

    def temperature(self, tr):
        """temper (im, jm, lm): interp3 of the archive's T over the read interval."""
        return self._interp3("t", "T", tr)

    def humidity(self, tr):
        """humid (im, jm, lm): interp3 of the archive's Q over the read interval."""
        return self._interp3("q", "Q", tr)

    def _to_grid(self, field):
        """A 1x1 surface field (360, 180) on the model grid: FillGrid 'area-aver' (a copy of
        the window for a 1x1 region)."""
        return fill(field, self.ll, ONE_BY_ONE)

    def _surface_record(self, param, t1, t2):
        """One record of a 1x1 surface field on the model grid (kept: callers do not write it)."""
        kept = self._surface.setdefault(param, {})
        if (t1, t2) not in kept:
            if len(kept) >= 4:
                kept.pop(next(iter(kept)))
            field = self._to_grid(self.archive.read(param, t1, t2)[param])
            field.flags.writeable = False
            kept[(t1, t2)] = field
        return kept[(t1, t2)]

    def surface_field(self, param, tr):
        """A 1x1 surface field (im, jm) over the read interval tr, by its TINTERP rule."""
        rule = TINTERP[param]
        if rule == "const":
            if param not in self._const:
                self._const[param] = self._to_grid(self.archive.read_const(param))
            return self._const[param]
        if rule == "month":
            return self._surface_record(param, *month(tr[0]))
        if rule == "aver1":
            return self._surface_record(param, *interval(tr[0], 1))
        tprev, tnext = instants(tr, 1)
        data = {t: self._surface_record(param, t, t) for t in (tprev, tnext)}
        return interpolate(tr, tprev, data[tprev], tnext, data[tnext])

    def surface(self, tr):
        """Calc_Kzz's surface fields over the read interval: {oro, lsm, sr, srols, u10m, v10m,
        sshf, slhf, ewss, nsss}, each (im, jm)."""
        return {param: self.surface_field(param, tr) for param in SURFACE}

    def kzz_inputs(self, tr, sp, pu, pv, tread=None):
        """The positional arguments of diffusion.calc_kzz for the step tr (Meteo_Setup_Other's
        tdyn .. tdyn + ndyn) in the read interval `tread` (tr if None), from the surface
        pressure `sp` advected to its start and its mass fluxes pu (im+1, jm, lm),
        pv (im, jm+1, lm) at the middle of the step."""
        a, b = self.levels(tr[0])
        return (
            self.spm(tr, tread), sp, pu, pv, self.temperature(tr), self.humidity(tr), self.surface(tr),
            a, b, self.dxyp, self.lmax_conv, self.ll.dlon, self.ll.dlat, self.ll.south,
        )  # fmt: skip

    def diffusion(self, tr, sp, pu, pv, tread=None):
        """Calc_Kzz: dkg (im, jm, lmax_conv) of the step (see kzz_inputs)."""
        dkg, _ = diffusion.calc_kzz(*self.kzz_inputs(tr, sp, pu, pv, tread))
        return dkg

    def convection(self, tr):
        """Meteo_Setup_Other_Core + Meteo_Produce_All for the convection: {entu, detu, entd,
        detd, cloud_top, cloud_lfs, cloud_base} of the 3-h interval containing tr[0]."""
        raw = self.archive.read("convec", *interval(tr[0]))
        eu, du, ed, dd = (raw[v][..., : self.lmax_conv] for v in VARIABLES["convec"])
        entu, detu, entd, detd = conv_unit_convert(eu, du, ed, dd, self.dxyp, self.czeta)
        top, lfs, base = cloud_dims(detu, entd)
        return {
            "entu": entu, "detu": detu, "entd": entd, "detd": detd,
            "cloud_top": top, "cloud_lfs": lfs, "cloud_base": base,
        }  # fmt: skip
