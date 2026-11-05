"""forward.share / unshare, how a reader process hands an interval over: nested dicts, lists and
tuples, large arrays (strided ones too) through one shared-memory block, small ones pickled;
a block given back is reused, one too small replaced."""

from datetime import datetime

import numpy as np

from pydyn import forward as F


def _interval(rng, n=40):
    big = rng.random((n, 30, 34))
    return {
        "tread": (datetime(2000, 1, 1), datetime(2000, 1, 1, 1)),
        "pairs": [(rng.random((n + 1, 30, 34)), rng.random((n, 31, 34)))],
        "steps": {"t": big, "cl": big[:, :, 3:], "a": rng.random(35), "loss": [{"k": 0.5}]},
    }


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def test_an_interval_crosses_whole_and_its_block_is_reused():
    rng = np.random.default_rng(3)
    first = _interval(rng)
    name, skeleton = F.share(first)
    try:
        assert name is not None and isinstance(skeleton["steps"]["t"], F._Shared)
        assert isinstance(skeleton["steps"]["a"], np.ndarray)  # small: pickled
        assert _same(F.unshare(name, skeleton), first)
        second = _interval(rng)
        again, skeleton = F.share(second, name)
        assert again == name  # large enough: reused
        assert _same(F.unshare(again, skeleton), second)
        larger = _interval(rng, 80)
        other, skeleton = F.share(larger, name)
        assert other != name  # outgrown: a new block
        assert _same(F.unshare(other, skeleton), larger)
    finally:
        F._free(name)
        F._free(locals().get("other"))


def test_a_reused_block_written_before_it_was_read_fails():
    """The known-bad: a block given out again before its interval was copied out."""
    rng = np.random.default_rng(4)
    first = _interval(rng)
    name, skeleton = F.share(first)
    try:
        F.share(_interval(rng), name)
        assert not _same(F.unshare(name, skeleton), first)
    finally:
        F._free(name)
