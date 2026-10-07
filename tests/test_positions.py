import base64
import gzip
import json
import re

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from zoommap import Transition, ZoomKeyframe, ZoomPositions


@pytest.fixture
def layouts():
    return [np.array([[-1., 0.], [0., 1.], [1., 0.]]),
            np.array([[-.5, 1.], [1., 0.], [.5, 1.]]),
            np.array([[0., 2.], [2., -1.], [0., 2.]])]


def test_nonuniform_zoom_grid_and_exact_endpoints(layouts):
    motion = ZoomPositions([(0, layouts[0]), (.25, layouts[1]), (3, layouts[2])])
    np.testing.assert_allclose(motion.positions_at(.125), (layouts[0] + layouts[1]) / 2)
    np.testing.assert_allclose(motion.positions_at(1.625), (layouts[1] + layouts[2]) / 2)
    for z, positions in zip(motion.zooms, layouts):
        np.testing.assert_array_equal(motion.positions_at(z), positions)
    np.testing.assert_array_equal(motion.positions_at(-100), layouts[0])
    np.testing.assert_array_equal(motion.positions_at(100), layouts[-1])


def test_reversal_has_no_history_and_caller_arrays_are_not_modified(layouts):
    original = [a.copy() for a in layouts]
    motion = ZoomPositions(list(zip([0, 1, 2], layouts)))
    forward = [motion.positions_at(z) for z in [.2, .7, 1.3]]
    backward = [motion.positions_at(z) for z in [1.3, .7, .2]][::-1]
    for a, b in zip(forward, backward):
        np.testing.assert_array_equal(a, b)
    layouts[0][:] = 999
    np.testing.assert_array_equal(motion.positions_at(0), original[0])
    evaluated = motion.positions_at(0)
    evaluated[:] = -999
    np.testing.assert_array_equal(motion.positions_at(0), original[0])


def test_per_interval_overrides(layouts):
    motion = ZoomPositions(list(zip([0, 1, 2], layouts)), transitions=[
        Transition(easing="smoothstep"), None,
    ])
    weight = .25 ** 2 * (3 - 2 * .25)
    np.testing.assert_allclose(motion.positions_at(.25), (1 - weight) * layouts[0] + weight * layouts[1])
    np.testing.assert_allclose(motion.positions_at(1.25), .75 * layouts[1] + .25 * layouts[2])


def test_pchip_uses_actual_zoom_spacing_and_stays_within_segment_bounds(layouts):
    knots = [0, .3, 2.7]
    motion = ZoomPositions(list(zip(knots, layouts)), interpolation="pchip")
    expected = PchipInterpolator(knots, np.stack(layouts), axis=0)
    for zoom in np.linspace(0, 2.7, 23):
        np.testing.assert_allclose(motion.positions_at(zoom), expected(zoom), atol=2e-6)
        index = min(int(np.searchsorted(knots, zoom, side="right") - 1), 1)
        low = np.minimum(layouts[index], layouts[index + 1])
        high = np.maximum(layouts[index], layouts[index + 1])
        assert (motion.positions_at(zoom) >= low - 2e-6).all()
        assert (motion.positions_at(zoom) <= high + 2e-6).all()


def test_binary_export_is_lossless_and_script_labels_are_safe(layouts):
    motion = ZoomPositions([ZoomKeyframe(0, layouts[0], '</script><script>alert("x")</script>'),
                            ZoomKeyframe(1, layouts[1], "local")])
    javascript = motion.to_javascript()
    assert "</script>" not in javascript
    specification = json.loads(re.search(r"const spec = (.*);", javascript).group(1))
    decoded = np.frombuffer(gzip.decompress(base64.b64decode(specification["positions"])), dtype="<f4")
    np.testing.assert_array_equal(decoded.reshape(2, 3, 2), np.stack(layouts[:2]))
    assert specification["labels"][0].startswith("</script>")


def test_plot_options_composes_custom_js_and_preserves_other_options(layouts):
    motion = ZoomPositions(list(zip([0, 1, 2], layouts)))
    options = motion.plot_options(custom_js="window.userCode = true;", height=900, title="Example")
    assert options["custom_js"].startswith("window.userCode = true;")
    assert options["height"] == 900 and options["title"] == "Example"


@pytest.mark.parametrize("knots", [[0, 0], [1, 0], [0, np.nan], [0, np.inf]])
def test_invalid_zooms_are_rejected(layouts, knots):
    with pytest.raises(ValueError):
        ZoomPositions(list(zip(knots, layouts)))


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.zeros(6), np.zeros((1, 2)),
                                  np.full((3, 2), np.nan), np.full((3, 2), np.inf),
                                  np.zeros((4, 2))])
def test_invalid_positions_are_rejected(layouts, bad):
    with pytest.raises(ValueError):
        ZoomPositions([(0, layouts[0]), (1, bad)])


def test_invalid_configuration_is_rejected(layouts):
    with pytest.raises(ValueError):
        ZoomPositions([(0, layouts[0])])
    with pytest.raises(ValueError):
        ZoomPositions([(0, np.zeros((3, 2))), (1, layouts[1])])
    with pytest.raises(ValueError):
        ZoomPositions(list(zip([0, 1, 2], layouts)), transitions=[Transition()])
    with pytest.raises(ValueError):
        ZoomPositions(list(zip([0, 1, 2], layouts)), zoom_reference="viewport")
    with pytest.raises(ValueError):
        Transition(interpolation="unknown")
    with pytest.raises(ValueError):
        Transition(easing="unknown")
    motion = ZoomPositions(list(zip([0, 1, 2], layouts)))
    with pytest.raises(ValueError):
        motion.positions_at(np.nan)
