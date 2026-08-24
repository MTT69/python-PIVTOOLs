"""Validation for caller-supplied destination buffers (``out=``).

Every reader that accepts ``out`` shares one contract: ``out`` is a C-contiguous
float32 array of exactly the shape the reader would otherwise allocate; it is
written in place and returned; after an exception its contents are undefined.
This module holds the one check that enforces it, so the wording is the same
whichever reader raises. It lives on the reader side of the import graph because
``load_images`` imports the readers, never the reverse.
"""

from typing import Tuple

import numpy as np


def check_out(out: np.ndarray, expected_shape: Tuple[int, ...], what: str) -> np.ndarray:
    """Return ``out`` if it satisfies the ``out=`` contract, else raise by name.

    Args:
        out: The caller's destination array.
        expected_shape: The exact shape the reader is about to produce.
        what: Who is checking, for the message (reader, camera, file).

    Raises:
        ValueError: On any shape, dtype or layout mismatch. The message lists
            every mismatch at once so one round trip fixes them all.
    """
    if not isinstance(out, np.ndarray):
        raise ValueError(f"{what}: out must be a numpy array, got {type(out).__name__}.")
    expected = tuple(int(n) for n in expected_shape)
    problems = []
    if tuple(out.shape) != expected:
        problems.append(f"shape {tuple(out.shape)}")
    if out.dtype != np.float32:
        problems.append(f"dtype {out.dtype}")
    if not out.flags.c_contiguous:
        problems.append("a non-C-contiguous layout")
    if problems:
        raise ValueError(
            f"{what}: out has {', '.join(problems)}; expected {expected} float32 "
            f"C-contiguous."
        )
    return out
