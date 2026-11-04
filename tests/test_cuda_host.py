"""CudaBackend's host side (backend.py) against a stand-in for cupy whose stream runs its
copies only when synchronized, as a GPU may: the uploads through pinned staging, the flags
deferred to the sync.  The kernels themselves need a GPU (test_gpu.py compiles them)."""

import ctypes
import sys
import types

import numpy as np
import pytest

from pydyn.gpu import backend as B


class _Stream:
    """One queue for every stream the backend makes (its own and the side one run apart only in
    that a settled read does not wait)."""

    def __init__(self):
        self.queue = []

    def synchronize(self, upto=None):
        while self.queue and self.queue[0] is not upto:
            self.queue.pop(0)()
        if self.queue:
            self.queue.pop(0)

    def use(self):
        pass


class _Event:
    def __init__(self, stream):
        self.stream = stream

    def record(self, stream):
        stream.queue.append(self)

    def synchronize(self):
        if self in self.stream.queue:
            self.stream.synchronize(self)


class _Memory:
    def __init__(self, arr, stream):
        self.arr, self.stream = arr, stream

    def copy_from_host_async(self, mem, size, stream):
        def run():  # the host bytes as they are when the copy runs, not when it was queued
            src = np.frombuffer((ctypes.c_uint8 * size).from_address(mem.value), np.uint8)
            self.arr.reshape(-1).view(np.uint8)[:size] = src

        stream.queue.append(run)


class _Array:
    def __init__(self, arr, stream):
        self.arr, self.stream = arr, stream
        self.data = _Memory(arr, stream)
        self.shape, self.dtype = arr.shape, arr.dtype

    def ravel(self):
        return _Array(self.arr.reshape(-1), self.stream)

    @property
    def flags(self):
        return self.arr.flags

    def get(self, stream=None):  # the side stream: nothing queued ahead of it
        return self.arr.copy()

    def __getitem__(self, key):
        return _Array(self.arr[key], self.stream)


def _fake_cupy():
    stream = _Stream()
    cp = types.ModuleType("cupy")
    cp.ndarray = _Array
    cp.empty = lambda shape, dtype: _Array(np.full(shape, -1, dtype), stream)
    cp.concatenate = lambda xs: _Array(np.concatenate([x.arr for x in xs]), stream)

    def asnumpy(a):
        stream.synchronize()
        return a.arr.copy()

    cp.asnumpy = asnumpy
    cp.cuda = types.SimpleNamespace(
        alloc_pinned_memory=bytearray,
        Stream=lambda non_blocking=False: stream,
        Event=lambda disable_timing=False: _Event(stream),
        runtime=types.SimpleNamespace(is_hip=False),
    )
    return cp, stream


@pytest.fixture
def cuda(monkeypatch):
    cp, stream = _fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(B._Staging, "CHUNK", 4096)
    return B.CudaBackend(), stream


def _uploads_survive(be, stream):
    """Uploads queued between syncs, then a sync: every one arrives as it was."""
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(3):
        host = [rng.random(n) for n in (5, 300, 1, 700, 37)]  # 700 doubles: a chunk of its own
        dev = [be.to_device(h) for h in host]
        assert stream.queue, "the uploads did not wait for the stream"
        be.sync()
        ok &= all(np.array_equal(d.arr, h) for d, h in zip(dev, host))
    return ok


def test_uploads_go_through_staging_until_the_sync(cuda):
    be, stream = cuda
    assert _uploads_survive(be, stream)
    assert len(be._staging.chunks) == 2  # reused after each sync: 3 rounds, no growth


def test_a_staging_block_reused_before_the_sync_fails(cuda, monkeypatch):
    be, stream = cuda

    def take_first(self, nbytes, tag, retired):  # the known-bad: no block held until its copy ran
        if not self.chunks:
            mem = bytearray(1 << 16)
            self.chunks.append([mem, np.frombuffer(mem, np.uint8), 0])
        return self.chunks[0][1][:nbytes]

    monkeypatch.setattr(B._Staging, "take", take_first)
    assert not _uploads_survive(be, stream)


def test_a_deferred_flag_raises_at_the_sync(cuda):
    be, _ = cuda
    clear, set_ = be.to_device(np.zeros(1, np.int32)), be.to_device(np.ones(1, np.int32))
    be.check(clear, ValueError("clear"), deferred=True)
    be.sync()
    be.check(clear, ValueError("clear"), deferred=True)
    be.check(set_, ValueError("set"), deferred=True)
    with pytest.raises(ValueError, match="set"):
        be.sync()
    be.sync()  # raised once


def test_a_download_waits_and_raises_what_was_deferred(cuda):
    be, _ = cuda
    x = be.to_device(np.arange(4.0))
    be.check(be.to_device(np.ones(1, np.int32)), ValueError("set"), deferred=True)
    with pytest.raises(ValueError, match="set"):
        be.to_host(x)
    assert np.array_equal(be.to_host(x), np.arange(4.0))


def test_a_mark_waits_for_what_came_before_it_only(cuda):
    """zoom.Pipeline's order: what is queued after a mark still waits when the mark is waited
    for; a settled read of what came before it does not sync."""
    be, stream = cuda
    a = be.to_device(np.arange(3.0))
    m = be.mark()
    b = be.to_device(np.arange(3.0) + 10)
    be.wait(m)
    assert np.array_equal(a.arr, np.arange(3.0))
    assert not np.array_equal(b.arr, np.arange(3.0) + 10), "the wait ran what came after the mark"
    be.settled = True
    try:
        assert np.array_equal(be.to_host(a), np.arange(3.0))
    finally:
        be.settled = False
    assert stream.queue, "the settled read synced"
    assert np.array_equal(be.to_host(b), np.arange(3.0) + 10)


def test_staging_waits_for_the_mark_of_its_copy(cuda):
    """A chunk's memory is reused only once the mark after its last copy has been waited for."""
    be, _ = cuda
    rng = np.random.default_rng(2)
    host = [rng.random(200) for _ in range(4)]  # two to a 4 kB chunk
    dev = [be.to_device(host[0]), be.to_device(host[1])]
    m = be.mark()
    dev.append(be.to_device(host[2]))  # the first chunk is in use: a second
    assert len(be._staging.chunks) == 2
    be.wait(m)
    dev.append(be.to_device(host[3]))  # the first chunk's copies have run: reused
    assert len(be._staging.chunks) == 2
    be.sync()
    assert all(np.array_equal(d.arr, h) for d, h in zip(dev, host))
