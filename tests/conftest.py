# tests/conftest.py
"""Shared fixtures for tests."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tracker():
    """Fresh BoardTracker at starting position."""
    # Import directly to avoid pulling in langgraph via __init__.py
    from rx200_agent.board_tracker import BoardTracker
    return BoardTracker()


@pytest.fixture
def starting_occupancy():
    """Standard starting position occupancy."""
    files = "abcdefgh"
    occupied = set()
    for f in files:
        occupied.add(f"{f}1")  # White back rank
        occupied.add(f"{f}2")  # White pawns
        occupied.add(f"{f}7")  # Black pawns
        occupied.add(f"{f}8")  # Black back rank
    return occupied


@pytest.fixture
def initial_state():
    """Fresh AgentState from create_initial_state."""
    # Import directly to avoid pulling in langgraph via __init__.py
    from rx200_agent.state import create_initial_state
    return create_initial_state(robot_color="black")
