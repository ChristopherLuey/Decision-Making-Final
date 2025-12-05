from __future__ import annotations

import math

from study.data.preprocess import _bounding_box, _primitive_parameters


def test_bounding_box_square():
    primitive = {
        "points": [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 10.0, "y": 5.0},
            {"x": 0.0, "y": 5.0},
        ]
    }
    bbox = _bounding_box([primitive])
    assert math.isclose(bbox["width"], 10.0)
    assert math.isclose(bbox["height"], 5.0)


def test_primitive_parameters_radius():
    primitive = {
        "type": "Circle",
        "center": {"x": 2.0, "y": 3.0},
        "radius": 1.5,
    }
    params = _primitive_parameters(primitive)
    assert math.isclose(params[0], 2.0)
    assert math.isclose(params[1], 3.0)
    assert math.isclose(params[4], 1.5)
