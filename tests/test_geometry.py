"""Tests for herald.scene.common.geometry."""

import numpy as np
import pytest

from herald.scene.common.geometry import (
    Frame,
    Geometry,
    Vec2,
    Vec3,
    Vec4,
)
from herald.scene.common.roi import ROI


@pytest.mark.parametrize(
    ("cls", "dim", "values", "expected"),
    [
        (Vec2, 2, (1.0, 2.0), [1.0, 2.0]),
        (Vec3, 3, (1.0, 2.0, 3.0), [1.0, 2.0, 3.0]),
        (Vec4, 4, (1.0, 2.0, 3.0, 4.0), [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_construct_from_sequence(cls, dim, values, expected):
    v = cls(values)
    assert isinstance(v, cls)
    assert isinstance(v, np.ndarray)
    assert v.shape == (dim,)
    assert v.dtype == np.float64
    assert np.allclose(v, expected)
    assert v.dim == dim


@pytest.mark.parametrize("cls", [Vec2, Vec3, Vec4])
def test_zeros(cls):
    v = cls.zeros()
    assert isinstance(v, cls)
    assert np.allclose(v, np.zeros(cls._dim))
    assert v.dim == cls._dim


def test_vec3_from_vec2_pads_zero():
    v2 = Vec2((1.0, 2.0))
    v3 = Vec3(v2)
    assert isinstance(v3, Vec3)
    assert np.allclose(v3, [1.0, 2.0, 0.0])


def test_vec2_from_vec3_trims():
    v3 = Vec3((1.0, 2.0, 3.0))
    v2 = Vec2(v3)
    assert isinstance(v2, Vec2)
    assert np.allclose(v2, [1.0, 2.0])


def test_vec4_from_vec3_pads_zero():
    v3 = Vec3((1.0, 2.0, 3.0))
    v4 = Vec4(v3)
    assert np.allclose(v4, [1.0, 2.0, 3.0, 0.0])


def test_vec4_from_vec2_pads_two_zeros():
    v2 = Vec2((4.0, 5.0))
    v4 = Vec4(v2)
    assert np.allclose(v4, [4.0, 5.0, 0.0, 0.0])


def test_vec3_from_vec4_trims():
    v4 = Vec4((1.0, 2.0, 3.0, 4.0))
    v3 = Vec3(v4)
    assert np.allclose(v3, [1.0, 2.0, 3.0])


def test_vec2_from_vec4_trims_to_xy():
    v4 = Vec4((1.0, 2.0, 3.0, 4.0))
    v2 = Vec2(v4)
    assert np.allclose(v2, [1.0, 2.0])


def test_vec3_from_short_sequence_pads():
    v3 = Vec3((7.0, 8.0))
    assert np.allclose(v3, [7.0, 8.0, 0.0])


def test_vec3_from_empty_sequence_is_zero():
    v3 = Vec3([])
    assert np.allclose(v3, [0.0, 0.0, 0.0])


def test_vector_arithmetic():
    v = Vec3((1.0, 2.0, 3.0))
    assert np.allclose(v + Vec3((0.5, 0.5, 0.5)), [1.5, 2.5, 3.5])
    assert np.allclose(v * 2.0, [2.0, 4.0, 6.0])


def test_tolist_round_trip():
    v = Vec3((1.0, 2.0, 3.0))
    assert v.tolist() == [1.0, 2.0, 3.0]
    assert np.allclose(Vec3(v.tolist()), v)


def test_geometry_defaults():
    geom = Geometry(type="point")
    assert geom.frame == "ENU"
    assert geom.coords.shape == (1, 3)
    assert np.allclose(geom.offset, [0.0, 0.0, 0.0])


def test_geometry_post_init_validates_coords():
    with pytest.raises(ValueError, match="coords must have shape"):
        Geometry(type="polygon", coords=np.zeros((2, 2)))


def test_geometry_to_dict_from_dict_round_trip():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 4.0, 5.0]],
        dtype=np.float64,
    )
    geom = Geometry(
        type="polygon",
        frame="ENU",
        coords=coords,
        offset=Vec3((0.0, 0.0, 12.0)),
    )

    payload = geom.to_dict()
    assert payload == {
        "type": "polygon",
        "frame": "ENU",
        "coords": coords.tolist(),
        "offset": [0.0, 0.0, 12.0],
    }

    loaded = Geometry.from_dict(payload)
    assert loaded.type == geom.type
    assert loaded.frame == geom.frame
    assert np.allclose(loaded.coords, geom.coords)
    assert np.allclose(loaded.offset, geom.offset)


def test_geometry_from_dict_pads_2d_coords():
    geom = Geometry.from_dict(
        {
            "type": "polyline",
            "frame": "WGS",
            "coords": [[48.71, 2.20], [48.72, 2.21]],
            "offset": [0.0, 0.0],
        }
    )
    assert geom.frame == "WGS"
    assert np.allclose(geom.coords, [[48.71, 2.20, 0.0], [48.72, 2.21, 0.0]])
    assert np.allclose(geom.offset, [0.0, 0.0, 0.0])


def test_frame_round_trip(tmp_path):
    frame = Frame.from_origin(48.71, 2.20)
    path = tmp_path / "frame.json"
    frame.save(path)
    loaded = Frame.load(path)
    assert loaded is not None
    assert loaded.lat == frame.lat
    assert loaded.lon == frame.lon
    assert loaded.east == 0.0
    assert loaded.north == 0.0


def test_frame_transforms_round_trip():
    frame = Frame.from_origin(48.71, 2.20)
    lat, lon = 48.711, 2.201
    east, north = frame.wgs2enu(lat, lon)
    lat2, lon2 = frame.enu2wgs(east, north)
    assert lat2 == pytest.approx(lat, abs=1e-9)
    assert lon2 == pytest.approx(lon, abs=1e-9)

    easting, northing = frame.wgs2utm(lat, lon)
    lat3, lon3 = frame.utm2wgs(easting, northing)
    assert lat3 == pytest.approx(lat, abs=1e-6)
    assert lon3 == pytest.approx(lon, abs=1e-6)

    east2, north2 = frame.utm2enu(easting, northing)
    assert east2 == pytest.approx(east, abs=1e-3)
    assert north2 == pytest.approx(north, abs=1e-3)

    easting2, northing2 = frame.enu2utm(east, north)
    assert easting2 == pytest.approx(easting, abs=1e-3)
    assert northing2 == pytest.approx(northing, abs=1e-3)


def test_frame_load_from_roi_centroid(tmp_path):
    roi = ROI.from_polygon(
        [(48.710, 2.200), (48.710, 2.203), (48.713, 2.203), (48.713, 2.200)]
    )
    path = tmp_path / "frame.json"
    lat, lon = roi.latlon_centroid()
    frame = Frame.from_origin(lat, lon)
    frame.save(path)
    loaded = Frame.load(path)
    assert loaded is not None
    assert loaded.lat == pytest.approx(lat)
    assert loaded.lon == pytest.approx(lon)
