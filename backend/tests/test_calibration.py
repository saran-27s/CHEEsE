"""Calibration geometry, square mapping and all four a1 orientations.

Images here are synthetic test fixtures drawn with OpenCV, not real photographs.
"""

import cv2
import numpy as np
import pytest

from chessvision.core.calibration import CalibrationError, CalibrationInput, build_calibration, validate_corners
from chessvision.core.coords import parse_square, square_at, square_center

W, H = 800, 600
# A perspective quadrilateral, clockwise on screen starting top-left.
QUAD = [(250.0, 120.0), (560.0, 110.0), (690.0, 480.0), (110.0, 500.0)]


def _point_near(corner, centroid, t=0.08):
    return (corner[0] + (centroid[0] - corner[0]) * t, corner[1] + (centroid[1] - corner[1]) * t)


@pytest.mark.parametrize("a1_index", [0, 1, 2, 3])
def test_each_orientation_maps_corners_to_expected_squares(a1_index):
    cal = build_calibration(CalibrationInput(corners=QUAD, a1_index=a1_index), W, H)
    centroid = (sum(p[0] for p in QUAD) / 4, sum(p[1] for p in QUAD) / 4)
    expected = ["a1", "a8", "h8", "h1"]  # clockwise from the a1 corner
    for j, name in enumerate(expected):
        corner = QUAD[(a1_index + j) % 4]
        bx, by = cal.image_to_board(*_point_near(corner, centroid))
        assert square_at(bx, by, cal.size) == parse_square(name), (a1_index, name)


@pytest.mark.parametrize("a1_index", [0, 1, 2, 3])
def test_warped_marker_lands_on_a1(a1_index):
    # Draw an 8x8 board into the quad and paint the a1 square red, then warp back.
    size = 640
    canonical = np.zeros((size, size, 3), np.uint8)
    cell = size // 8
    for f in range(8):
        for r in range(8):
            color = (200, 200, 200) if (f + r) % 2 else (60, 60, 60)
            canonical[r * cell:(r + 1) * cell, f * cell:(f + 1) * cell] = color
    canonical[7 * cell:8 * cell, 0:cell] = (0, 0, 255)  # a1 in canonical frame (bottom-left)
    src_canon = np.float32([(0, size), (0, 0), (size, 0), (size, size)])  # a1, a8, h8, h1
    dst_img = np.float32([QUAD[(a1_index + j) % 4] for j in range(4)])
    to_image = cv2.getPerspectiveTransform(src_canon, dst_img)
    image = cv2.warpPerspective(canonical, to_image, (W, H))

    cal = build_calibration(CalibrationInput(corners=QUAD, a1_index=a1_index), W, H, size=size)
    warped = cal.warp(image)
    cx, cy = square_center(parse_square("a1"), size)
    b, g, r = warped[int(cy), int(cx)]
    assert r > 200 and g < 60 and b < 60
    hx, hy = square_center(parse_square("h8"), size)
    assert not (warped[int(hy), int(hx)][2] > 200 and warped[int(hy), int(hx)][1] < 60)


def test_square_at_does_not_clamp():
    assert square_at(-1, 10, 640) is None
    assert square_at(640, 10, 640) is None
    assert square_at(0, 0, 640) == parse_square("a8")
    assert square_at(639, 639, 640) == parse_square("h1")


def test_counter_clockwise_rejected():
    errors = validate_corners(list(reversed(QUAD)), W, H)
    assert any("counter-clockwise" in e for e in errors)


def test_self_intersecting_rejected():
    bowtie = [QUAD[0], QUAD[2], QUAD[1], QUAD[3]]
    errors = validate_corners(bowtie, W, H)
    assert any("convex" in e for e in errors)


def test_tiny_and_outside_rejected():
    tiny = [(10, 10), (12, 10), (12, 12), (10, 12)]
    assert any("area" in e for e in validate_corners(tiny, W, H))
    outside = [(-200, 0), (500, 0), (500, 500), (0, 500)]
    assert any("outside" in e for e in validate_corners(outside, W, H))


def test_build_raises_with_all_errors():
    with pytest.raises(CalibrationError) as exc:
        build_calibration(CalibrationInput(corners=list(reversed(QUAD)), a1_index=0), W, H)
    assert exc.value.errors


def test_camera_dir_points_toward_image_bottom_for_standard_view():
    # a1 at bottom-left on screen => White side nearest the camera => camera is toward rank 1 (down).
    cal = build_calibration(CalibrationInput(corners=QUAD, a1_index=3), W, H)
    assert cal.camera_dir is not None
    assert cal.camera_dir[1] > 0.5


def test_camera_dir_none_for_prewarped_square_image():
    # An already-warped (top-down) image has no foreshortening, so pieces are anchored at box centre.
    square = [(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)]
    cal = build_calibration(CalibrationInput(corners=square, a1_index=3), 640, 640)
    assert cal.camera_dir is None
    assert cal.to_dict()["camera_dir_source"].startswith("none_detected")


@pytest.mark.parametrize("a1_index,expected", [(3, (0.0, 1.0)), (1, (0.0, -1.0)), (0, (1.0, 0.0)), (2, (-1.0, 0.0))])
def test_camera_dir_follows_orientation(a1_index, expected):
    # QUAD's bottom edge on screen is the longest, so the camera is on the screen-bottom side.
    # Which chess side that is depends on where a1 was clicked:
    #   a1_index 3 -> bottom edge a1-h1 (rank 1, +y)   a1_index 1 -> bottom edge a8-h8 (rank 8, -y)
    #   a1_index 0 -> bottom edge h8-h1 (h-file, +x)   a1_index 2 -> bottom edge a1-a8 (a-file, -x)
    cal = build_calibration(CalibrationInput(corners=QUAD, a1_index=a1_index), W, H)
    dx, dy = cal.camera_dir
    assert dx * expected[0] + dy * expected[1] > 0.7


def test_camera_side_override():
    cal = build_calibration(CalibrationInput(corners=QUAD, a1_index=3, camera_side="rank8"), W, H)
    assert cal.camera_dir == (0.0, -1.0) and cal.camera_dir_source == "user: rank8"
    cal2 = build_calibration(CalibrationInput(corners=QUAD, a1_index=3, camera_side="none"), W, H)
    assert cal2.camera_dir is None
