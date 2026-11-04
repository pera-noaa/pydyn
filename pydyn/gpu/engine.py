"""The device engine of the replay (`replay --device emu|cuda`): replay.HostEngine's members for
a ZoomState on the device.  The state is uploaded when seeded from a dump or an input file;
the operators run through pydyn.gpu.zoom; a comparison downloads the one array it compares
("verify"); the host is told the sub-cycle counts and the flags ("control").  Nothing else of
the state comes down: Calc_Kzz, the loss rates and every operator run where the state is."""

from ..harness import compare
from ..replay import _sha1
from . import kernels as G
from . import zoom as gz
from .backend import set_backend
from .diffusion import calc_kzz

__all__ = ["DeviceEngine"]


class DeviceEngine:
    fingerprint = False

    def __init__(self, device):
        self.backend = set_backend(device)
        self.name = self.backend.name

    @property
    def ledger(self):
        return self.backend.ledger

    def zeros(self, zm, nt):
        return gz.zeros(zm, nt)

    def copy_into(self, view, arr):
        for lo, hi, alo, ahi in zip(view.lower, view.upper, arr.lower, arr.upper):
            if lo < alo or hi > ahi:
                raise ValueError(f"dump bounds {arr} do not cover {view}")
        view.data[...] = self.backend.to_device(arr.data[arr.window(view)], "dump")

    def put(self, view, values):
        self.backend.put(view.data, values)

    def host(self, x, tag="verify"):
        return self.backend.to_host(x, tag) if self.backend.is_device(x) else x

    def compare(self, view, arr, name):
        ours = self.backend.to_host(view.data, "verify")
        report = compare(arr.data[arr.window(view)], ours, name)
        if self.fingerprint:
            report["sha1"] = _sha1(ours)
        return report

    def report(self, model, ours, name):
        ours = self.host(ours)
        report = compare(model, ours, name)
        if self.fingerprint:
            report["sha1"] = _sha1(ours)
        return report

    calc_kzz = staticmethod(calc_kzz)

    @staticmethod
    def column_pressure(zm, m, dxyp):
        return gz._column_pressure(zm, m, dxyp)

    def chemistry(self, inputs, dt_sec):
        return G.loss_rates(dt_sec, **inputs)

    @property
    def asynchronous(self):
        return self.backend.asynchronous

    def sync(self):
        self.backend.sync()

    def pipeline(self, zm, st, depth):
        return gz.Pipeline(zm, st, depth)
