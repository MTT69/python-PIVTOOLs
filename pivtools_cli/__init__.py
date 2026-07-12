"""
PIVTOOLs - Particle Image Velocimetry Tools
"""

try:
    from importlib.metadata import version

    __version__ = version("pivtools")
except Exception:
    __version__ = "0.5.1"
