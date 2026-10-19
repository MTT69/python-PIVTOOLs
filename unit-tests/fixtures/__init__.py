"""Shared builders for calibration tests: real-layout vector sets and stale-file makers.

These are Python helpers rather than data files because every ``.mat`` under
``unit-tests/`` is gitignored. Import as ``from fixtures.vector_sets import ...``
(``unit-tests/`` is on ``sys.path`` under pytest's default import mode).
"""
