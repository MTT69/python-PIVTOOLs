"""
PIVTOOLs Core - Shared utilities for PIVTOOLs packages
"""

try:
    from importlib.metadata import version
    __version__ = version("pivtools")
except Exception:
    __version__ = "0.4.5"
