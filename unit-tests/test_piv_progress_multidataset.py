"""Progress reporting for multi-dataset PIV runs.

Pins the regression that motivated this module: with several source/base path pairs
selected, the GUI progress bar reached 100% as soon as the *first* dataset finished
because the percentage was scoped to a single base path. See
``pivtools_gui.app.get_uncalibrated_count``.
"""

import os
import time

import pytest

from pivtools_core.paths import count_vector_files_by_name, get_data_paths
from pivtools_gui.app import (
    _scan_dataset_progress,
    _settled_vector_files,
    _tail_lines,
)

NUM_PAIRS = 4
CAMERAS = [1, 2]
TYPE_NAME = "instantaneous"
PER_DATASET = NUM_PAIRS * len(CAMERAS)
EXPECTED_NAMES = {f"B{i:05d}.mat" for i in range(1, NUM_PAIRS + 1)}


def _data_dir(base, cam):
    """Uncalibrated output directory for one camera of one dataset."""
    return get_data_paths(
        base, NUM_PAIRS, cam, TYPE_NAME, use_uncalibrated=True
    )["data_dir"]


def _write_results(base, cam, count, age_s=0.0):
    """Create ``count`` result files for one camera, optionally aged."""
    folder = _data_dir(base, cam)
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(EXPECTED_NAMES)[:count]
    for name in names:
        path = folder / name
        path.write_bytes(b"x")
        if age_s:
            old = time.time() - age_s
            os.utime(path, (old, old))
    return folder


def _make_datasets(tmp_path, n):
    bases = []
    for i in range(n):
        base = tmp_path / f"dataset_{i}"
        base.mkdir()
        bases.append(base)
    return bases


def _scan(bases, active):
    return _scan_dataset_progress(
        bases, active, CAMERAS, NUM_PAIRS, TYPE_NAME, EXPECTED_NAMES, PER_DATASET
    )


# --------------------------------------------------------------------------
# count_vector_files_by_name / _settled_vector_files
# --------------------------------------------------------------------------


def test_count_missing_directory_is_zero(tmp_path):
    """A dataset that has not started has no output folder -- not an error."""
    assert count_vector_files_by_name(tmp_path / "nope", EXPECTED_NAMES) == 0


def test_count_ignores_unexpected_names(tmp_path):
    folder = _write_results(tmp_path, 1, 2)
    (folder / "notes.txt").write_bytes(b"x")
    (folder / "B99999.mat").write_bytes(b"x")
    assert count_vector_files_by_name(folder, EXPECTED_NAMES) == 2


def test_count_is_bounded_by_expected_names(tmp_path):
    """The membership filter caps the count, so percent can never exceed 100."""
    folder = _write_results(tmp_path, 1, NUM_PAIRS)
    for extra in range(100, 110):
        (folder / f"B{extra:05d}.mat").write_bytes(b"x")
    assert count_vector_files_by_name(folder, EXPECTED_NAMES) == NUM_PAIRS


def test_counting_ignores_file_age(tmp_path):
    """Counting must NOT apply the settle guard.

    Applying it delayed the progress bar by the settle window for no benefit --
    counting never opens the file.
    """
    folder = _write_results(tmp_path, 1, 3, age_s=0.0)
    assert count_vector_files_by_name(folder, EXPECTED_NAMES) == 3


def test_settled_excludes_young_files(tmp_path):
    """Status-image candidates keep the settle guard -- they do get read."""
    folder = _write_results(tmp_path, 1, 2, age_s=600.0)
    fresh = folder / sorted(EXPECTED_NAMES)[2]
    fresh.write_bytes(b"x")

    settled = _settled_vector_files(folder, EXPECTED_NAMES, time.time())

    assert len(settled) == 2
    assert fresh.name not in settled


def test_settled_missing_directory_is_empty(tmp_path):
    assert _settled_vector_files(tmp_path / "nope", EXPECTED_NAMES, time.time()) == []


# --------------------------------------------------------------------------
# _scan_dataset_progress
# --------------------------------------------------------------------------


def test_nothing_started(tmp_path):
    bases = _make_datasets(tmp_path, 4)
    found, complete, idx, position = _scan(bases, [0, 1, 2, 3])
    assert (found, complete) == (0, 0)
    assert (idx, position) == (0, 1)


def test_first_dataset_complete_is_not_the_whole_run(tmp_path):
    """The reported regression.

    Dataset 0 finished; the other three have not started. Progress must read 25%,
    not 100%, and the run must point at dataset 1.
    """
    bases = _make_datasets(tmp_path, 4)
    for cam in CAMERAS:
        _write_results(bases[0], cam, NUM_PAIRS)

    found, complete, idx, position = _scan(bases, [0, 1, 2, 3])

    total_expected = PER_DATASET * 4
    assert found == PER_DATASET
    assert complete == 1
    assert (idx, position) == (1, 2)
    assert int(found / total_expected * 100) == 25


def test_partial_second_dataset(tmp_path):
    bases = _make_datasets(tmp_path, 4)
    for cam in CAMERAS:
        _write_results(bases[0], cam, NUM_PAIRS)
    _write_results(bases[1], CAMERAS[0], NUM_PAIRS)
    _write_results(bases[1], CAMERAS[1], 2)

    found, complete, idx, position = _scan(bases, [0, 1, 2, 3])

    assert found == PER_DATASET + NUM_PAIRS + 2
    assert complete == 1
    assert (idx, position) == (1, 2)


def test_all_complete_has_no_current_dataset(tmp_path):
    """No current dataset once everything is done -- callers must render no label."""
    bases = _make_datasets(tmp_path, 3)
    for base in bases:
        for cam in CAMERAS:
            _write_results(base, cam, NUM_PAIRS)

    found, complete, idx, position = _scan(bases, [0, 1, 2])

    assert found == PER_DATASET * 3
    assert complete == 3
    assert idx is None and position is None
    assert int(found / (PER_DATASET * 3) * 100) == 100


def test_subset_selection_only_counts_selected(tmp_path):
    """Unselected datasets must not appear in numerator or denominator."""
    bases = _make_datasets(tmp_path, 4)
    for cam in CAMERAS:
        _write_results(bases[3], cam, NUM_PAIRS)

    found, complete, idx, position = _scan(bases, [1, 3])

    assert found == PER_DATASET
    assert complete == 1
    assert (idx, position) == (1, 1)


def test_processing_order_is_honoured(tmp_path):
    """active_paths order defines which dataset is 'current', not numeric order."""
    bases = _make_datasets(tmp_path, 3)
    for cam in CAMERAS:
        _write_results(bases[2], cam, NUM_PAIRS)

    _, _, idx, position = _scan(bases, [2, 0, 1])

    assert (idx, position) == (0, 2)


# --------------------------------------------------------------------------
# _tail_lines
# --------------------------------------------------------------------------


def _oracle(path, count):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()[-count:] if count > 0 else []


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("".join(f"line {i}\n" for i in range(10)), id="plain"),
        pytest.param(
            "".join(f"line {i}\n" for i in range(9)) + "line 9", id="no-trailing-nl"
        ),
        pytest.param("", id="empty"),
        pytest.param("only\n", id="single"),
        pytest.param("a\n\n\nb\n", id="blank-lines"),
        pytest.param(
            "".join(f"{i} " + "x" * 60 + "\n" for i in range(20000)), id="multi-chunk"
        ),
    ],
)
@pytest.mark.parametrize("count", [0, 1, 3, 500, 20001])
def test_tail_matches_text_mode_readlines(tmp_path, content, count):
    """Tail read must be indistinguishable from reading the whole file.

    Including newline translation: the log is written CRLF on Windows and the old
    whole-file path normalised it via text mode.
    """
    log = tmp_path / "job.log"
    log.write_text(content, encoding="utf-8")
    assert _tail_lines(log, count) == _oracle(log, count)


def test_tail_reads_far_less_than_the_file(tmp_path):
    """The point of the tail read: cost must not scale with run length."""
    log = tmp_path / "big.log"
    log.write_text("".join(f"{i} " + "y" * 200 + "\n" for i in range(100000)))
    size = log.stat().st_size

    read_bytes = []
    real_open = open

    class _Counting:
        def __init__(self, handle):
            self._handle = handle

        def read(self, n=-1):
            data = self._handle.read(n)
            read_bytes.append(len(data))
            return data

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

    import builtins

    builtins.open = lambda *a, **k: _Counting(real_open(*a, **k))
    try:
        lines = _tail_lines(log, 500)
    finally:
        builtins.open = real_open

    assert len(lines) == 500
    assert sum(read_bytes) < size / 10
