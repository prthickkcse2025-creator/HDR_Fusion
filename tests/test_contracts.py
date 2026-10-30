import numpy as np
import pytest
from hdrfusion.contracts import assert_native_resolution, assert_finite
from hdrfusion.tiling import make_tiles, overlap_add

def test_native_resolution():
    assert_native_resolution((4480, 6720, 3), (4480, 6720, 3))
    with pytest.raises(ValueError):
        assert_native_resolution((2240, 3360, 3), (4480, 6720, 3))

def test_finite():
    assert_finite(np.ones((3, 3), np.float32))
    with pytest.raises(ValueError):
        assert_finite(np.array([np.inf]))

def test_overlap_add_identity():
    rng = np.random.default_rng(4)
    image = rng.random((91, 137, 3), dtype=np.float32)
    tiles = make_tiles(91, 137, 48, 16)
    rebuilt = overlap_add([image[t.y0:t.y1, t.x0:t.x1] for t in tiles], tiles, image.shape)
    assert rebuilt.shape == image.shape
    assert np.max(np.abs(rebuilt - image)) < 1e-5
