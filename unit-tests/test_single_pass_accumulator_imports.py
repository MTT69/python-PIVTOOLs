"""
Regression test for name-shadowing inside finalize_pass (2026-07-13).

A `from pathlib import Path` / `import os` near the bottom of finalize_pass
(store_planes block) made both names function-local for the ENTIRE ~1000-line
function body, so the save_fit_diagnostics block earlier in the function
raised `UnboundLocalError: cannot access local variable 'Path'` on every
ensemble run with diagnostics enabled. Both are now module-level imports;
this guards against the in-function imports being reintroduced.
"""

from pivtools_cli.piv.piv_backend.single_pass_accumulator import (
    SinglePassAccumulator,
)


def test_finalize_pass_does_not_shadow_module_level_imports():
    local_names = SinglePassAccumulator.finalize_pass.__code__.co_varnames
    assert "Path" not in local_names, (
        "finalize_pass contains a local `from pathlib import Path`, which "
        "shadows the module-level import for the whole function and breaks "
        "earlier uses of Path (UnboundLocalError)"
    )
    assert "os" not in local_names, (
        "finalize_pass contains a local `import os`, which shadows the "
        "module-level import for the whole function"
    )
