"""A serializable position sequence; no imports of DataMapPlot internals."""

from dataclasses import dataclass
from importlib.resources import files
import base64
import gzip
import json

import numpy as np


@dataclass(frozen=True)
class Transition:
    """Spatial interpolation and easing for one interval between keyframes.

    PCHIP uses every keyframe to estimate derivatives on the actual zoom grid.
    Easing is applied within an interval; smoothstep intentionally stops at knots.
    """

    interpolation: str = "linear"
    easing: str = "linear"

    def __post_init__(self):
        if self.interpolation not in {"linear", "pchip"}:
            raise ValueError("interpolation must be 'linear' or 'pchip'")
        if self.easing not in {"linear", "smoothstep", "smootherstep"}:
            raise ValueError("easing must be 'linear', 'smoothstep', or 'smootherstep'")


@dataclass(frozen=True)
class ZoomKeyframe:
    zoom: float
    positions: np.ndarray
    label: str = ""


def _encode_floats(array):
    raw = np.asarray(array, dtype="<f4").tobytes(order="C")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def _ease(t, easing):
    if easing == "smoothstep":
        return t * t * (3 - 2 * t)
    if easing == "smootherstep":
        return t * t * t * (t * (6 * t - 15) + 10)
    return t


class ZoomPositions:
    """Ordered (zoom, positions) keyframes with stable row identity.

    ``positions`` has shape (N, 2) at every keyframe. All coordinates must
    already share a display coordinate system. The first frame must be the
    array passed as DataMapPlot's first positional argument. Browser setup
    verifies that agreement and learns DataMapPlot's normalization from it.

    ``zoom_reference='initial'`` means offsets from the plot's initial fitted
    zoom. ``'absolute'`` uses deck.gl zoom values directly. Values outside the
    sequence hold its endpoint positions, without constraining the camera.

    ``transitions`` optionally contains K-1 Transition objects (or None to
    inherit the defaults). Point IDs are their row indices in every frame.
    """

    def __init__(self, keyframes, *, zoom_reference="initial",
                 interpolation="linear", easing="linear", transitions=None):
        if zoom_reference not in {"initial", "absolute"}:
            raise ValueError("zoom_reference must be 'initial' or 'absolute'")
        default = Transition(interpolation, easing)
        prepared = []
        shape = None
        for frame in keyframes:
            if not isinstance(frame, ZoomKeyframe):
                frame = ZoomKeyframe(*frame)
            zoom = float(frame.zoom)
            positions = np.array(frame.positions, dtype=np.float32, copy=True)
            if not np.isfinite(zoom):
                raise ValueError("keyframe zooms must be finite")
            if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
                raise ValueError("each positions array must have shape (N, 2), N >= 2")
            if not np.isfinite(positions).all():
                raise ValueError("positions must contain only finite coordinates")
            if shape is not None and positions.shape != shape:
                raise ValueError("all keyframes must have the same shape and row order")
            if prepared and zoom <= prepared[-1].zoom:
                raise ValueError("keyframe zooms must be strictly increasing")
            if not isinstance(frame.label, str):
                raise ValueError("keyframe labels must be strings")
            positions.setflags(write=False)
            prepared.append(ZoomKeyframe(zoom, positions, frame.label))
            shape = positions.shape
        if len(prepared) < 2:
            raise ValueError("at least two position keyframes are required")
        if np.ptp(prepared[0].positions, axis=0).max() == 0:
            raise ValueError("the first keyframe must have nonzero extent")
        if transitions is None:
            transitions = [default] * (len(prepared) - 1)
        else:
            transitions = list(transitions)
            if len(transitions) != len(prepared) - 1:
                raise ValueError("transitions must have one entry per keyframe interval")
            transitions = [default if t is None else t for t in transitions]
            if not all(isinstance(t, Transition) for t in transitions):
                raise ValueError("transition entries must be Transition objects or None")
        self.keyframes = tuple(prepared)
        self.transitions = tuple(transitions)
        self.zoom_reference = zoom_reference
        self.zooms = tuple(frame.zoom for frame in self.keyframes)
        self._positions = np.stack([frame.positions for frame in self.keyframes])
        self._positions.setflags(write=False)
        self._coefficients = None
        if any(t.interpolation == "pchip" for t in self.transitions):
            from scipy.interpolate import PchipInterpolator
            # Shape (interval, polynomial degree, point, xy), highest degree first.
            self._coefficients = np.moveaxis(
                PchipInterpolator(self.zooms, self._positions, axis=0).c, 0, 1
            ).astype(np.float32)

    def positions_at(self, zoom):
        """Evaluate in the supplied keyframes' coordinate system, for research/QA."""
        zoom = float(zoom)
        if not np.isfinite(zoom):
            raise ValueError("zoom must be finite")
        if zoom <= self.zooms[0]:
            return self._positions[0].copy()
        if zoom >= self.zooms[-1]:
            return self._positions[-1].copy()
        i = int(np.searchsorted(self.zooms, zoom, side="right") - 1)
        width = self.zooms[i + 1] - self.zooms[i]
        transition = self.transitions[i]
        t = _ease((zoom - self.zooms[i]) / width, transition.easing)
        if transition.interpolation == "pchip":
            a, b, c, d = self._coefficients[i]
            u = t * width
            return ((a * u + b) * u + c) * u + d
        return (1 - t) * self._positions[i] + t * self._positions[i + 1]

    def to_javascript(self):
        """Generate custom_js, including compressed Float32 coordinate data.

        The script never patches a DataMapPlot class or changes its call API.
        It subscribes to view events and replaces only getPosition on the
        existing point layer. The exported HTML needs no Python process.
        """
        specification = {
            "zooms": self.zooms,
            "labels": [frame.label for frame in self.keyframes],
            "reference": self.zoom_reference,
            "count": self._positions.shape[1],
            "positions": _encode_floats(self._positions),
            "transitions": [
                {"interpolation": t.interpolation, "easing": t.easing}
                for t in self.transitions
            ],
            "coefficients": _encode_floats(self._coefficients)
            if self._coefficients is not None else None,
        }
        # Labels are data, including when a label happens to contain </script>.
        encoded = json.dumps(specification, allow_nan=False).replace("<", "\\u003c")
        template = files("zoommap").joinpath("positions.js").read_text(encoding="utf-8")
        return template.replace("__ZOOMMAP_SPEC__", encoded)

    def plot_options(self, **options):
        """Compose with existing options, including an existing custom_js string."""
        options = dict(options)
        options["custom_js"] = (options.get("custom_js") or "") + "\n" + self.to_javascript()
        return options
