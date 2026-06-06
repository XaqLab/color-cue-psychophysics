import numpy as np
from color_cue.stimulus import (
    cue_to_theta,
    get_cmap,
    get_theta_array,
    theta_to_cue,
)


def test_cue_theta_roundtrip():
    cues = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    theta = cue_to_theta(cues)
    recovered = theta_to_cue(theta)
    np.testing.assert_allclose(recovered, cues)


def test_theta_to_cue_clips():
    recovered = theta_to_cue(np.array([-10.0, 10.0]))
    np.testing.assert_allclose(recovered, np.array([0.0, 1.0]))


def test_cmap_loads():
    cmap = get_cmap()
    assert cmap.N > 0


def test_theta_array_shape():
    arr = get_theta_array(-np.pi / 4, size=(16, 16), rng=0)
    assert arr.shape == (16, 16)
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)
