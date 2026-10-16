"""Perception: synthetic segmentation, backprojection accuracy, state format."""
import numpy as np
import pytest

from harness.perception import (ObjectReading, format_object_lines, perceive,
                                to_json)


def _synthetic_scene():
    """32x32 red square at rows/cols 8..20, flat depth 500mm, identity-ish camera."""
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[8:20, 8:20] = (200, 30, 30)  # red-ish
    depth = np.full((32, 32), 200.0, np.float32)  # millimeters (0.2 m tabletop)
    K = np.array([[32.0, 0, 16.0], [0, 32.0, 16.0], [0, 0, 1.0]])
    # cam2world_gl = GL2CV  =>  world == CV point (GL2CV @ GL2CV == I)
    G2W = np.diag([1.0, -1.0, -1.0, 1.0])
    return rgb, depth, K, G2W


def _sp(K, G2W):
    return {"base_camera": {"intrinsic_cv": K[None], "cam2world_gl": G2W[None]}}


def test_segment_and_unproject():
    rgb, depth, K, G2W = _synthetic_scene()
    rs = perceive(rgb, depth, None, None, _sp(K, G2W), "PickCube-v1")
    assert len(rs) == 1 and rs[0].name == "cube"
    r = rs[0]
    assert r.camera == "base_camera" and r.mask_pixels >= 100
    # centroid ~ (13.5, 13.5) px, depth 0.5 m -> x=y=(13.5-16)/32*0.5
    assert r.position is not None
    assert abs(r.position[2] - 0.2) < 1e-6
    assert abs(r.position[0] - (13.5 - 16) * 0.2 / 32) < 0.01
    assert abs(r.position[1] - (13.5 - 16) * 0.2 / 32) < 0.01


def test_not_visible_when_no_match():
    rgb = np.zeros((32, 32, 3), np.uint8)
    depth = np.full((32, 32), 200.0, np.float32)
    _, _, K, G2W = _synthetic_scene()
    rs = perceive(rgb, depth, None, None, _sp(K, G2W), "PickCube-v1")
    assert rs[0].position is None and rs[0].camera == "none"


def test_wrist_priority_by_pixel_count():
    rgb, depth, K, G2W = _synthetic_scene()
    big = rgb.copy(); big[4:28, 4:28] = (200, 30, 30)   # 576 px wrist view
    small_depth = np.full((32, 32), 200.0, np.float32)
    sp = {"base_camera": {"intrinsic_cv": K[None], "cam2world_gl": G2W[None]},
          "hand_camera": {"intrinsic_cv": K[None], "cam2world_gl": G2W[None]}}
    rs = perceive(rgb, small_depth, big, np.full((32, 32), 200.0, np.float32), sp, "PickCube-v1")
    assert rs[0].camera == "hand_camera"


def test_format_lines_visible_and_not():
    ee = np.array([0.0, 0.0, 0.0])
    vis = ObjectReading("cube", (0.1, 0.0, 0.2), "hand_camera", 512)
    lines = format_object_lines([vis, ObjectReading("peg", None, "none", 0)], ee)
    assert "object cube: center pos=(0.100, 0.000, 0.200) m" in lines[0]
    assert "dist=0.224 m" in lines[0]
    assert "not visible" in lines[1]


def test_to_json():
    rs = [ObjectReading("cube", (0.1, 0.0, 0.2), "base_camera", 40),
          ObjectReading("peg", None, "none", 0)]
    j = to_json(rs)
    assert j["cube"]["pos"] == [0.1, 0.0, 0.2] and j["cube"]["camera"] == "base_camera"
    assert j["peg"]["pos"] is None


def test_last_known_position_memory():
    ee = np.array([0.0, 0.0, 0.0])
    stale = ObjectReading("cube", None, "none", 0,
                          last_seen_pos=(0.1, 0.0, 0.2), last_seen_age=7)
    lines = format_object_lines([stale], ee)
    assert "not visible now; last seen 7 step(s) ago" in lines[0]
    assert "pos=(0.100, 0.000, 0.200) m" in lines[0]
    assert "dist=0.224 m" in lines[0]
    j = to_json([stale])
    assert j["cube"]["pos"] is None
    assert j["cube"]["last_seen_pos"] == [0.1, 0.0, 0.2]
    assert j["cube"]["last_seen_age_steps"] == 7
    # no memory -> plain not-visible line, no last_seen keys
    blind = ObjectReading("peg", None, "none", 0)
    assert "not visible in either camera" in format_object_lines([blind], ee)[0]
    assert "last_seen_pos" not in to_json([blind])["peg"]
