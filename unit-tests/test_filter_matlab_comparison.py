"""
Test POD filter and time filter against MATLAB reference output.

Prerequisites:
    1. Open MATLAB, cd to the unit-tests/ directory
    2. Run: generate_filter_reference
    3. This creates test_output/pod_reference.mat and test_output/time_reference.mat
    4. Then run: pytest test_filter_matlab_comparison.py -v
"""

import numpy as np
import pytest
import scipy.io as sio
from pathlib import Path

from pivtools_cli.preprocessing.pod_filter import (
    find_auto_mode,
    pod_filter_single_channel,
    pod_filter_batch,
    time_filter_batch,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "test_output"
POD_REF = FIXTURE_DIR / "pod_reference.mat"
TIME_REF = FIXTURE_DIR / "time_reference.mat"
