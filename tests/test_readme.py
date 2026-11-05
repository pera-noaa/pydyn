"""README.md's example runs as printed, and its two advections agree."""

import re
from pathlib import Path

import numpy as np

README = Path(__file__).parents[1] / "README.md"


def test_readme_example():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert len(blocks) == 1, "README.md should hold exactly one python example"
    scope = {}
    exec(blocks[0], scope)
    for name in ("m", "rm", "rxm", "rym", "rzm"):  # test_simple.py's criterion
        ref = getattr(scope["ref"], name)
        got = getattr(scope["got"], name)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
