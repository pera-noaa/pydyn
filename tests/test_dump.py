"""The dump reader (pydyn.dump)."""


import numpy as np
import pytest

from pydyn.dump import FArray


# ------------------------------------------------------------------ dump reader
def test_farray_uses_fortran_bounds():
    data = np.arange(4 * 3, dtype=float).reshape(4, 3, order="F")
    a = FArray(data, lower=(-1, 0))
    assert a.upper == (2, 2)
    assert a[-1, 0] == data[0, 0]
    assert a[2, 2] == data[3, 2]
    np.testing.assert_array_equal(a[0:1, 1:2], data[1:3, 1:3])
    np.testing.assert_array_equal(a[0:1], data[1:3])  # trailing axes whole
    assert a[0:1, 2].shape == (2,)
    with pytest.raises(IndexError):
        a[-1:2:2]


@pytest.mark.parametrize("text, why", [
    ("seq = 1\nregion = 1\nim = 2\njm = 2\nlm = 1\n", "no tag"),
    ("tag = meteo\nseq = 1\nregion = 1\nim = 2\njm = 2\nlm = 1\narray m = 1 2 1\n", "not an array line"),
    ("tag = meteo\nseq = 1\nregion = 1\nim = 2\njm = 2\nlm = 1\narray m = 5 1\n", "bad array bounds"),
    ("tag = meteo\nseq = 1\nregion = 1\nim = 2\njm = 2\nlm = 1\narray ../m = 1 2\n", "or name"),
])
def test_a_malformed_header_is_a_dump_error(tmp_path, text, why):
    """A truncated or foreign .meta is named, not a KeyError deep in the replay."""
    from pydyn import replay
    from pydyn.dump import DumpError, Event

    path = tmp_path / "e00001_meteo_r1_p000.meta"
    path.write_text(text)
    with pytest.raises(DumpError, match=why):
        Event(path)
    with pytest.raises(SystemExit) as stop:
        replay.main([str(tmp_path)])
    assert stop.value.code == 2
