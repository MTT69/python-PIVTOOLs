"""
Pytest configuration for PIVTools tests.
"""
import pytest


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--make-figures", action="store_true", default=False,
        help="Generate diagnostic figures from tests into tests/test_output/",
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
