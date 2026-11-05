"""The inputs a run reads from files: `RegionInputs` for one region, `Ingestion` one per region
of a tree (a single-region run is a tree of one) -- what the replay compares with the dumped
ones and, in a free run, uses.  Each source is optional:

    meteo       a meteo.Meteo (or zoom_meteo.ChildMeteo) over the region's archive: the raw
                mass fluxes, matched surface pressures and initial air mass of a `meteo`
                event, the convection fields of a bef_v and Calc_Kzz's dkg of its step,
                and the temperature of a chem event
    emissions   an emis.Emissions on the region: the month's field of every `emis` event
    loss        a chemfields.LossFields on the region: the loss-rate fields of a chem event
    initial     the istart-66 file (initial.initial_state on the region's grid)

Every method returns what it read and its comparison with the dump (`in:<name>`), the
arrays as the dump holds them (the whole region).  `scope` is the region's interior scope
(zoom.scope), where Calc_Kzz runs: a child's edge ring keeps dkg 0.  The caches hold one read
interval (the convection) and one dynamics step (dkg) per region.
"""

from datetime import timedelta

import numpy as np

from . import balance, chemfields, emis, initial, meteo, regrid
from . import zoom as Z
from . import zoom_meteo as ZM
from .harness import compare

__all__ = [
    "MASS_INPUTS", "Ingestion", "RegionInputs", "chem_month", "dumped_dkg", "midpoint", "read_interval",
    "seed", "times",
]  # fmt: skip

# the raw inputs of Meteo_Setup_Mass a meteo event carries, in the dump's names
MASS_INPUTS = ("mfu1", "mfv1", "mfw1", "mfu2", "mfv2", "mfw2", "sp1", "sp2")


def times(ev, k1, k2):
    """The pair of dump-header dates (k1, k2) as datetimes."""
    return balance.date(ev.meta[k1]), balance.date(ev.meta[k2])


def read_interval(ev, key="idate"):
    """The model's meteo read interval (tread1, tread2) at the event's time."""
    t = balance.date(ev.meta[key])
    t1 = t - timedelta(seconds=(t - t.replace(hour=0, minute=0, second=0)).total_seconds() % ev.meta["nread"])
    return t1, t1 + timedelta(seconds=ev.meta["nread"])


def midpoint(ev):
    """The middle of a chem event's step, whose month (and year) pick the climatologies."""
    t1, t2 = times(ev, "tr1", "tr2")
    return t1 + (t2 - t1) / 2


def chem_month(chem_ev):
    """The middle of a chem event's step, whose month (and year) pick the climatologies,
    checked against the dumped month."""
    t = midpoint(chem_ev)
    if chem_ev.meta["month"] != t.month:
        raise ValueError(f"chem: dumped month {chem_ev.meta['month']}, midpoint {t}")
    return t


def dumped_dkg(ev, lmax):
    """The dkg of a bef_v dump (zero when a without_diffusion build dumped none)."""
    im, jm = ev.meta["im"], ev.meta["jm"]
    return ev["dkg"][1:im, 1:jm, 1:lmax] if "dkg" in ev else np.zeros((im, jm, lmax))


class RegionInputs:
    def __init__(self, meteo=None, emissions=None, loss=None, initial=None, ll=None, scope=None):
        self.meteo, self.emissions, self.loss, self.initial = meteo, emissions, loss, initial
        self.ll, self.scope = ll, scope
        self._conv, self._dkg = {}, {}

    def mass(self, ev):
        """Meteo_Setup_Mass's inputs of a meteo (or cfl) event: the raw fluxes and matched
        surface pressures of its read interval, and on the run's first the initial sp, phlb
        and m.  A cfl event's sp is Check_CFL's, recomputed from the restored mass: not compared."""
        out = self.meteo.mass(times(ev, "tread1", "tread2"))
        names = [n for n in (*MASS_INPUTS, "sp", "m") if n in out and (n, ev.tag) != ("sp", "cfl")]
        return out, [compare(ev.interior(n), out[n], f"in:{n}") for n in names]

    def convection(self, ev, step=None, engine=None):
        """The convection inputs of a bef_v event -- entu..detd and the cloud levels of its
        read interval, and given the `step` (the step interval, its mid-step pu, pv and its
        starting sp) Calc_Kzz's dkg over the scope -- as kernels.convec kwargs, and reports.
        An `engine` (replay.HostEngine or pydyn.gpu's) computes and reports the dkg itself."""
        im, jm = ev.meta["im"], ev.meta["jm"]
        lmax = ev["entu"].upper[2]
        self.meteo.lmax_conv = lmax
        tr = read_interval(ev)
        if tr not in self._conv:
            self._conv = {tr: self.meteo.convection(tr)}
        out = self._conv[tr]
        reports = [compare(ev[name].data, q, f"in:{name}") for name, q in out.items()]
        conv = {name: q[:im, :jm] for name, q in out.items() if name != "cloud_base"}
        if step is not None:
            if step["tr"] not in self._dkg:
                if engine is None:
                    dkg = self.meteo.diffusion(step["tr"], step["sp"], step["pu"], step["pv"], tr)
                else:
                    args = self.meteo.kzz_inputs(step["tr"], step["sp"], step["pu"], step["pv"], tr)
                    dkg = engine.calc_kzz(*args)
                if self.scope is not None:  # Calc_Kzz runs on isr:ier x jsr:jer
                    inside, dkg = dkg[self.scope], np.zeros_like(dkg)
                    dkg[self.scope] = inside
                self._dkg = {step["tr"]: dkg}
            conv["dkg"] = self._dkg[step["tr"]]
            if engine is None:
                reports.append(compare(dumped_dkg(ev, lmax), conv["dkg"], "in:dkg"))
            else:
                reports.append(engine.report(dumped_dkg(ev, lmax), conv["dkg"], "in:dkg"))
        return conv, reports

    def temperature(self, chem_ev, tr):
        """The temperature of a chem event over Meteo_Setup_Other's interval `tr` (the step's,
        tdyn .. tdyn + ndyn): (temper, report)."""
        temper = self.meteo.temperature(tr)
        im, jm, lm = chem_ev.meta["im"], chem_ev.meta["jm"], chem_ev.meta["lm"]
        return temper, compare(chem_ev["temper"][1:im, 1:jm, 1:lm], temper, "in:temper")

    def chemistry(self, chem_ev, tr=None):
        """The inputs of a chem event read from the files -- the temperature over `tr` (with
        a meteo), the loss-rate fields (with a loss) -- and their reports: (dict or None, reports)."""
        out, reports = {}, []
        if self.meteo is not None and tr is not None:
            out["temper"], report = self.temperature(chem_ev, tr)
            reports.append(report)
        if self.loss is not None:
            meta, t = chem_ev.meta, chem_month(chem_ev)
            fields = self.loss.month(t.month, t.year, meta["trop_cl_sink"], meta["soil_sink"])
            reports += [compare(chem_ev[name].data, q, f"in:{name}") for name, q in fields.items()]
            out.update(fields)
        return out or None, reports

    def sources(self, inner):
        """The do_add_2d calls of an s pair's `emis` events from the month's field and the
        calendar: ([(x, l_level, i_tracer)], the field against each dumped one)."""
        injections, reports = [], []
        for e in inner:
            m = e.meta
            t = balance.date(m["idate"])
            reports.append(compare(e["emis"].data, self.emissions.field(t), "in:emis"))
            x = self.emissions.injection(t, m["nsrce"], m["mol_mass"], m["mol_mass_e"], m["efrac"], m["tref"])
            injections.append((x, m["l_level"], m["i_tracer"]))
        return injections, reports

    def initial_state(self, m0, a, b, nt):
        """The istart-66 tracers {rm, rxm, rym, rzm} of the region on its initial air mass m0
        and half levels (a, b)."""
        kw = {} if self.ll is None else {"ll": self.ll}
        return initial.initial_state(self.initial, m0, a, b, emis.XMCH4, nt, **kw)


class Ingestion:
    """The inputs of every region read from the files the model reads, one RegionInputs per
    region (`ingestion[r]`), each source optional:

    `meteo_root`   the tm5-nc archive: by default for a tree its 1x1, 137-level set regridded
                   onto each region (regrid.RegriddedArchive), as the zoom rc reads it, and for
                   one region its set on the model grid (meteo.Archive), as ch4_6x4_fwd.rc reads
                   it (`meteo_grid` 'model' / '1x1'); meteo.Meteo for region 1,
                   zoom_meteo.ChildMeteo for a child
    `emissions`    the 1x1 monthly CH4 files, each region's field on its own grid (emission_ct's
                   coarsen onto lli(region))
    `chemistry`    the CH4 input tree, each region's loss-rate fields (read_OH_field, the soil
                   map per region)
    `initial`      the istart-66 file, Fill3D onto each region, then update_parent

    Calc_Kzz runs on each region's interior scope."""

    def __init__(self, zm, meteo_root=None, emissions=None, chemistry=None, initial=None, meteo_grid=None):
        self.zm, self.initial, self.has_meteo = zm, initial, meteo_root is not None
        meteo_grid = meteo_grid or ("model" if zm.nregions == 1 else "1x1")
        if meteo_grid == "model" and zm.nregions > 1:
            raise NotImplementedError("Ingestion: a zoom reads the 1x1 set (meteo_grid='1x1')")
        self.region, source = {}, None
        for r in zm.regions():
            g = zm.g(r)
            met = None
            if meteo_root is not None:
                if meteo_grid == "model":  # the archive's own region set, on the model grid
                    archive = meteo.Archive(meteo_root)
                else:
                    archive = regrid.RegriddedArchive(meteo_root, g.ll, source)
                    source = archive.source
                parent = None if r == 1 else self.region[g.parent].meteo
                if r == 1:
                    met = meteo.Meteo(archive, g.ll)
                else:
                    met = ZM.ChildMeteo(archive, parent, ZM.Refinement.of(zm, r), g.ll)
            self.region[r] = RegionInputs(
                meteo=met,
                emissions=None if emissions is None else emis.Emissions(emissions, ll=g.ll),
                loss=None if chemistry is None else chemfields.LossFields(chemistry, g.ll, g.name),
                initial=initial, ll=g.ll, scope=Z.scope(zm, r),
            )  # fmt: skip

    def __getitem__(self, r):
        return self.region[r]

    def initial_state(self, m0, nt):
        """The run's start (initexit start, istart 66): each region's tracers from the file
        on its own grid and its own initial air mass `m0` {region: (im, jm, lm)}, then
        update_parent from the deepest region up.  A ZoomState."""
        zm = self.zm
        st = Z.ZoomState.zeros(zm, nt)
        for r in zm.regions():
            g = zm.g(r)
            ic = self.region[r].initial_state(m0[r], zm.at, zm.bt, nt)
            st[r]["m"][1 : g.im, 1 : g.jm, 1 : zm.lm] = m0[r]
            for k, q in ic.items():
                st[r][k][1 : g.im, 1 : g.jm, 1 : zm.lm] = q
        for r in sorted(zm.regions(), reverse=True):
            Z.update_parent(zm, st, r)
        return st


def seed(zm, st, ic, engine):
    """initexit's start on `st` (where `engine` keeps it) from Ingestion.initial_state's `ic`:
    every region's m and tracers, a cyclic region's m with its periodic x halos two deep
    (Pressure_to_Mass; update_parent leaves them), then each parent's m2phlb that ends
    update_parent.  Returns the parents (their sp now m2phlb's)."""
    for r in zm.regions():
        for k in ("m", "rm", "rxm", "rym", "rzm"):
            engine.put(st[r][k], ic[r][k].data)
        g, m = zm.g(r), st[r]["m"]
        if g.xcyc:
            m[-1:0, 1 : g.jm, :] = m[g.im - 1 : g.im, 1 : g.jm, :]
            m[g.im + 1 : g.im + 2, 1 : g.jm, :] = m[1:2, 1 : g.jm, :]
    parents = [r for r in zm.regions() if zm.children[r]]
    for r in parents:
        (st.lib.m2phlb1 if r == 1 else st.lib.m2phlb)(zm, st, r)
    return parents
