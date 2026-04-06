"""Tests for space.py and encoding.py."""

import math
from imads_hpo.space import Categorical, Integer, LogReal, Real, Space
from imads_hpo.encoding import SpaceEncoder


def test_real_roundtrip():
    space = Space({"x": Real(0.0, 10.0)})
    enc = SpaceEncoder(space, resolution=100)
    params = {"x": 5.0}
    coords = enc.encode(params)
    decoded = enc.decode(coords)
    assert abs(decoded["x"] - 5.0) < 0.15  # within 1 mesh step


def test_logreal_roundtrip():
    space = Space({"lr": LogReal(1e-5, 1e-1)})
    enc = SpaceEncoder(space, resolution=1000)
    params = {"lr": 1e-3}
    coords = enc.encode(params)
    decoded = enc.decode(coords)
    # Log-space rounding: should be within ~1 order of magnitude
    assert 0.5e-3 < decoded["lr"] < 2e-3


def test_integer_roundtrip():
    space = Space({"bs": Integer(16, 256, step=16)})
    enc = SpaceEncoder(space)
    params = {"bs": 64}
    coords = enc.encode(params)
    decoded = enc.decode(coords)
    assert decoded["bs"] == 64


def test_categorical_roundtrip():
    space = Space({"opt": Categorical(["adam", "sgd", "adamw"])})
    enc = SpaceEncoder(space)
    for choice in ["adam", "sgd", "adamw"]:
        params = {"opt": choice}
        coords = enc.encode(params)
        decoded = enc.decode(coords)
        assert decoded["opt"] == choice


def test_categorical_uses_int64():
    space = Space({"opt": Categorical(["a", "b", "c"])})
    enc = SpaceEncoder(space)
    coords = enc.encode({"opt": "b"})
    assert coords == [1]
    assert isinstance(coords[0], int)


def test_mixed_space():
    space = Space({
        "lr": LogReal(1e-5, 1e-1),
        "bs": Integer(16, 256, step=16),
        "opt": Categorical(["adam", "sgd"]),
        "dropout": Real(0.0, 0.5),
    })
    enc = SpaceEncoder(space)
    assert enc.search_dim == 4

    params = {"lr": 1e-3, "bs": 64, "opt": "sgd", "dropout": 0.2}
    coords = enc.encode(params)
    assert len(coords) == 4

    decoded = enc.decode(coords)
    assert decoded["opt"] == "sgd"
    assert decoded["bs"] == 64


def test_mesh_base_step_is_positive():
    space = Space({"x": Real(0, 1), "y": Integer(0, 10)})
    enc = SpaceEncoder(space)
    assert enc.mesh_base_step > 0
