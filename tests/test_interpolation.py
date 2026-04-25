# tests/test_interpolation.py
"""
Tests for bilinear interpolation of 64 square positions from 4 corners.

Tests the pure math — does NOT require FastAPI or robot hardware.
"""

import pytest


def bilinear_interpolate(corners, files="abcdefgh", ranks="12345678"):
    """
    Pure-Python bilinear interpolation from 4 corners to 64 squares.

    This mirrors the logic in POST /square_positions/interpolate.

    Args:
        corners: dict with keys a1, a8, h1, h8 each having x, y, z, pitch
    Returns:
        dict of 64 squares -> {x, y, z, pitch}
    """
    c_a1 = corners["a1"]
    c_a8 = corners["a8"]
    c_h1 = corners["h1"]
    c_h8 = corners["h8"]

    result = {}
    for fi, f in enumerate(files):
        u = fi / 7.0
        for ri, r in enumerate(ranks):
            v = ri / 7.0
            sq = f + r
            pos = {}
            for key in ["x", "y", "z"]:
                val_a1 = c_a1.get(key, 0)
                val_a8 = c_a8.get(key, 0)
                val_h1 = c_h1.get(key, 0)
                val_h8 = c_h8.get(key, 0)
                bottom = val_a1 * (1 - u) + val_h1 * u
                top = val_a8 * (1 - u) + val_h8 * u
                pos[key] = bottom * (1 - v) + top * v
            pitches = [c.get("pitch", 90) for c in [c_a1, c_a8, c_h1, c_h8]]
            pos["pitch"] = sum(pitches) / len(pitches)
            result[sq] = pos
    return result


@pytest.fixture
def simple_corners():
    """Corners forming a unit square in XY, flat Z."""
    return {
        "a1": {"x": 0.0, "y": 0.0, "z": 0.05, "pitch": 90},
        "a8": {"x": 0.0, "y": 0.35, "z": 0.05, "pitch": 90},
        "h1": {"x": 0.35, "y": 0.0, "z": 0.05, "pitch": 90},
        "h8": {"x": 0.35, "y": 0.35, "z": 0.05, "pitch": 90},
    }


@pytest.fixture
def tilted_corners():
    """Corners with Z tilt (board not perfectly flat)."""
    return {
        "a1": {"x": 0.0, "y": 0.0, "z": 0.05, "pitch": 90},
        "a8": {"x": 0.0, "y": 0.35, "z": 0.06, "pitch": 90},
        "h1": {"x": 0.35, "y": 0.0, "z": 0.04, "pitch": 90},
        "h8": {"x": 0.35, "y": 0.35, "z": 0.07, "pitch": 90},
    }


class TestCornerPositions:
    """Verify corner positions are preserved exactly."""

    def test_a1_preserved(self, simple_corners):
        result = bilinear_interpolate(simple_corners)
        assert abs(result["a1"]["x"] - 0.0) < 1e-9
        assert abs(result["a1"]["y"] - 0.0) < 1e-9

    def test_h8_preserved(self, simple_corners):
        result = bilinear_interpolate(simple_corners)
        assert abs(result["h8"]["x"] - 0.35) < 1e-9
        assert abs(result["h8"]["y"] - 0.35) < 1e-9

    def test_a8_preserved(self, simple_corners):
        result = bilinear_interpolate(simple_corners)
        assert abs(result["a8"]["x"] - 0.0) < 1e-9
        assert abs(result["a8"]["y"] - 0.35) < 1e-9

    def test_h1_preserved(self, simple_corners):
        result = bilinear_interpolate(simple_corners)
        assert abs(result["h1"]["x"] - 0.35) < 1e-9
        assert abs(result["h1"]["y"] - 0.0) < 1e-9


class TestInterpolation:
    """Verify interpolated positions are correct."""

    def test_64_squares_generated(self, simple_corners):
        result = bilinear_interpolate(simple_corners)
        assert len(result) == 64

    def test_all_square_names_valid(self, simple_corners):
        result = bilinear_interpolate(simple_corners)
        files = "abcdefgh"
        ranks = "12345678"
        for f in files:
            for r in ranks:
                assert f + r in result

    def test_center_position(self, simple_corners):
        """d4/e5 area should be roughly at (0.175, 0.175)."""
        result = bilinear_interpolate(simple_corners)
        # d-file = index 3 => u = 3/7, rank 4 = index 3 => v = 3/7
        d4 = result["d4"]
        expected_x = 0.35 * (3.0 / 7.0)
        expected_y = 0.35 * (3.0 / 7.0)
        assert abs(d4["x"] - expected_x) < 1e-9
        assert abs(d4["y"] - expected_y) < 1e-9

    def test_e4_midboard(self, simple_corners):
        """e4: file e=index 4 => u=4/7, rank 4=index 3 => v=3/7."""
        result = bilinear_interpolate(simple_corners)
        e4 = result["e4"]
        expected_x = 0.35 * (4.0 / 7.0)
        expected_y = 0.35 * (3.0 / 7.0)
        assert abs(e4["x"] - expected_x) < 1e-9
        assert abs(e4["y"] - expected_y) < 1e-9

    def test_flat_z_all_equal(self, simple_corners):
        """With flat corners, all Z values should be 0.05."""
        result = bilinear_interpolate(simple_corners)
        for sq, pos in result.items():
            assert abs(pos["z"] - 0.05) < 1e-9, f"{sq} z={pos['z']}"

    def test_tilted_z_interpolates(self, tilted_corners):
        """With tilted corners, Z should vary across the board."""
        result = bilinear_interpolate(tilted_corners)
        # a1 z=0.05, h8 z=0.07 — center should be in between
        z_vals = [pos["z"] for pos in result.values()]
        assert min(z_vals) >= 0.04 - 1e-9
        assert max(z_vals) <= 0.07 + 1e-9
        # d4 should be intermediate
        d4_z = result["d4"]["z"]
        assert 0.04 < d4_z < 0.07

    def test_pitch_averaged(self, simple_corners):
        """Pitch should be the average of all 4 corners."""
        result = bilinear_interpolate(simple_corners)
        for sq, pos in result.items():
            assert abs(pos["pitch"] - 90.0) < 1e-9


class TestEdgeCases:
    """Edge cases for the interpolation."""

    def test_edge_squares_b1_between_a1_h1(self, simple_corners):
        """b1 is on rank 1, file b (index 1) => u=1/7."""
        result = bilinear_interpolate(simple_corners)
        b1 = result["b1"]
        expected_x = 0.35 * (1.0 / 7.0)
        assert abs(b1["x"] - expected_x) < 1e-9
        assert abs(b1["y"] - 0.0) < 1e-9

    def test_monotonic_x_along_files(self, simple_corners):
        """X should increase monotonically from a-file to h-file."""
        result = bilinear_interpolate(simple_corners)
        for r in "12345678":
            prev_x = -999
            for f in "abcdefgh":
                x = result[f + r]["x"]
                assert x > prev_x, f"X not monotonic at {f}{r}"
                prev_x = x

    def test_monotonic_y_along_ranks(self, simple_corners):
        """Y should increase monotonically from rank 1 to rank 8."""
        result = bilinear_interpolate(simple_corners)
        for f in "abcdefgh":
            prev_y = -999
            for r in "12345678":
                y = result[f + r]["y"]
                assert y > prev_y, f"Y not monotonic at {f}{r}"
                prev_y = y
