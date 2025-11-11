#include <Python.h>

// This file creates a minimal, valid Python C-extension.
// Its only purpose is to define the `PyInit__build_marker` function
// to satisfy the linker and force setuptools to build a platform-specific wheel.

// 1. Define an empty method table
static PyMethodDef BuildMarkerMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// 2. Define the module structure
static struct PyModuleDef buildmarkermodule = {
    PyModuleDef_HEAD_INIT,
    "pivtools_cli._build_marker", // Module name
    NULL,                         // Module docstring
    -1,                           // Module state size (no state)
    BuildMarkerMethods            // Method table
};

// 3. Define the module initialization function
// This is the function the linker was looking for.
PyMODINIT_FUNC PyInit__build_marker(void) {
    return PyModule_Create(&buildmarkermodule);
}