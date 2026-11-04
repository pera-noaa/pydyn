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
