"""pivtools_gui.services — neutral, app-wide service utilities.

Shared infrastructure that is not specific to any one subsystem. ``job_manager`` (the
background-job tracker) lives here because it is used across calibration, video_maker,
transforms, and plotting; it was relocated out of the legacy ``calibration/`` package as
part of the v1 retirement so no subsystem depends on ``calibration/`` for it.
"""
