# Tests

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_board_tracker.py -v

# Single test class
python -m pytest tests/test_board_tracker.py::TestCastling -v

# With coverage
python -m pytest tests/ --cov=rx200_agent --cov-report=term-missing
```

## Test Structure

- `conftest.py` — Shared fixtures (`tracker`, `starting_occupancy`, `initial_state`)
- `test_board_tracker.py` — BoardTracker unit tests (moves, captures, castling, en passant, promotion, detection, serialization)
- `test_game_flow.py` — Integration test: Scholar's Mate game flow + state routing

## Requirements

```
pytest
python-chess
```

No hardware or Docker required — all tests use pure Python logic.
