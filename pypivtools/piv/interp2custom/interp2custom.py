import ctypes
import os

import numpy as np

KERNEL_LANCZOS = 0
KERNEL_GAUSSIAN = 1


class Interp2Custom:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Use platform-appropriate library extension
            lib_extension = ".dll" if os.name == "nt" else ".so"
            lib_path = os.path.join(
                os.path.dirname(__file__), "../lib", f"libinterp2custom{lib_extension}"
            )
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))

        # Set C function signatures
        self.lib.interp2custom.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # y
            np.ctypeslib.ndpointer(dtype=np.uintp, flags="F_CONTIGUOUS"),  # N
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # f_i
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # f_j
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # yi (output)
            ctypes.c_int,  # n_interp
        ]
        self.lib.interp2custom.restype = None

    def interpolate(
        self, y, i, j, kernel_type="lanczos", kernel_size=4, gaussian_std=0.65
    ):
        # -------------------
        # Validate inputs
        # -------------------
        if y.ndim != 2:
            raise ValueError("y must be a 2D array")
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        if i.shape != j.shape:
            raise ValueError("i and j must have the same shape")
        if i.dtype != np.float32:
            i = i.astype(np.float32)
        if j.dtype != np.float32:
            j = j.astype(np.float32)

        # Kernel type as int
        if kernel_type.lower() == "lanczos":
            ktype = KERNEL_LANCZOS
        elif kernel_type.lower() == "gaussian":
            ktype = KERNEL_GAUSSIAN
        else:
            raise ValueError(f"Unknown kernel type '{kernel_type}'")

        self.lib.interp1custom_generatelut(
            ktype, kernel_size, ctypes.c_float(gaussian_std)
        )

        N = np.asfortranarray([y.shape[0], y.shape[1]], dtype=np.uintp)
        n_interp = i.size
        yi = np.asfortranarray(np.zeros_like(i, dtype=np.float32))
        y = np.asfortranarray(y)
        i = np.asfortranarray(i)
        j = np.asfortranarray(j)
        self.lib.interp2custom(y, N, i, j, yi, n_interp)
        return yi
