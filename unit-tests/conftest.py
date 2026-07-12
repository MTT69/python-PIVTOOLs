"""
Pytest configuration for PIVTOOLs unit tests.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to sys.path so production code can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--make-figures",
        action="store_true",
        default=False,
        help="Generate diagnostic figures from tests into unit-tests/test_output/",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def make_figures(request):
    """Return True when --make-figures was passed on the CLI."""
    return request.config.getoption("--make-figures")


@pytest.fixture
def output_dir():
    """Return (and create) the test output directory."""
    d = Path(__file__).resolve().parent / "test_output"
    d.mkdir(exist_ok=True)
    return d
