"""analysis/cfl_cap_replay_zoom.py's plumbing: the tree it builds from a dump header is the
tree, a header that disagrees with the Python is refused, and the census of an interval is
zoom.check_cfl_zoom's decision.  Fake events stand in for a dump set: the real check is the
control column on a real one, which needs Orion's glb3x2 + nam1x1 dumps."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_zoom import production, three, two

from pydyn import synth
from pydyn import zoom as Z
from pydyn.dump import FArray, _parse_value

_spec = importlib.util.spec_from_file_location(
    "cfl_cap_replay_zoom", Path(__file__).resolve().parents[1] / "analysis" / "cfl_cap_replay_zoom.py")
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)

T1, T2 = (2019, 1, 1, 0, 0, 0), (2019, 1, 1, 1, 0, 0)


class FakeEvent:
    tag = "cfl"

    def __init__(self, region, meta, arrays):
        self.region, self.meta, self._arrays, self.prefix = region, meta, arrays, f"fake_r{region}"

    def __getitem__(self, name):
        return self._arrays[name]


def _events(zm, st, ndyn, n, nread):
    """One interval's cfl events as the zoom header writes them, pu1 = pu2 = the state's."""
    evs = {}
    for r in zm.regions():
        g, f = zm.g(r), st[r]
        meta = {
            "im": g.im, "jm": g.jm, "lm": zm.lm, "grid_name": g.name, "grid_reduced": False, "nred": 0,
            "xref": g.xref, "yref": g.yref, "zref": g.zref, "tref": g.tref, "parent": g.parent,
            "xcyc": g.xcyc, "touch_sp": g.touch_sp, "touch_np": g.touch_np,
            "ndyn_in": ndyn, "n_in": n, "nread": nread, "ndyn_max": nread,
            "tread1": T1, "pu_t1": T1, "pu_t2": T2,
            "status": 0, "splitorder": zm.splitorderzoom[r],
            # as dump.py parses the line: one child is an int, none an empty tuple
            "children": _parse_value(" ".join(str(c) for c in zm.children[r])),
        }  # fmt: skip
        if r > 1:
            meta.update({k: getattr(zm, k)[r] for k in ("ibeg", "iend", "jbeg", "jend")})
        arrays = {k: FArray(f[k].data.copy(), f[k].lower) for k in ("m", "sp", "dxyp")}
        for k in ("pu", "pv"):
            arrays[f"{k}1"] = arrays[f"{k}2"] = FArray(f[k].data.copy(), f[k].lower)
        arrays["at"], arrays["bt"] = FArray(zm.at.copy(), (1,)), FArray(zm.bt.copy(), (1,))
        evs[r] = FakeEvent(r, meta, arrays)
    return evs


@pytest.mark.parametrize("tree", [two, three, production], ids=lambda f: f.__name__)
def test_tree_from_header(tree):
    zm = tree()
    evs = _events(zm, synth.zoom_meteo(zm, np.random.default_rng(3)), 3600, 1, 3600)
    got = census.build_tree(evs)
    assert (got.ibeg, got.iend, got.jbeg, got.jend) == (zm.ibeg, zm.iend, zm.jbeg, zm.jend)
    assert got.splitorderzoom == zm.splitorderzoom
    assert {r: sorted(c) for r, c in got.children.items()} == {r: sorted(c) for r, c in zm.children.items()}


def test_header_disagreement_is_refused():
    zm = two()
    evs = _events(zm, synth.zoom_meteo(zm, np.random.default_rng(3)), 3600, 1, 3600)
    got = census.build_tree(evs)
    evs[2].meta["ibeg"] += 1
    with pytest.raises(SystemExit, match="ibeg"):
        census.check_tree(got, evs)
    evs[2].meta["ibeg"] -= 1
    evs[2].meta["splitorder"] = evs[2].meta["splitorder"][::-1] + "x"
    with pytest.raises(SystemExit, match="splitorder"):
        census.check_tree(got, evs)


@pytest.mark.parametrize(("tree", "cfl", "caps"), [
    (two, 0.9, (10, 1, 1)), (two, 1.2, (10, 1, 1)), (two, 1.2, (10, 1, 2)),
    (production, 1.2, (10, 1, 1)), (production, 1.2, (10, 1, 2)),
], ids=lambda v: getattr(v, "__name__", str(v)))  # fmt: skip
def test_census_is_check_cfl_zoom(tree, cfl, caps):
    """Same decision as check_cfl_zoom on the state directly (the census interpolates
    pu1 = pu2, which may move a flux by an ulp, never a decision here)."""
    zm = tree()
    st = synth.zoom_meteo(zm, np.random.default_rng(7), cfl=cfl)
    evs = _events(zm, st, 3600, 1, 3600)
    want = Z.check_cfl_zoom(zm, st.copy(), 3600, 1, 3600, caps=caps)
    got = census.replay_interval(evs, census.build_tree(evs), caps)
    assert got[:2] == want[:2]
