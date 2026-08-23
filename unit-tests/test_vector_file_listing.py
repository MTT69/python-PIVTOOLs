"""Directory scanning for expected PIV result files.

Pins the behaviour that replaced the per-frame ``Path.exists()`` sweep in the GUI
backend. The sweep cost one syscall per frame -- 1037 ms against 5 ms for a single
``os.scandir`` on a 3600-pair dataset -- and was the dominant cost of opening the
Results tab. See ``pivtools_core.paths.list_vector_files``.

The ordering tests are the load-bearing ones: every call site takes ``[0]`` expecting
the *lowest* frame, which the old ``range(1, n + 1)`` loop gave for free and
``os.scandir`` does not.
"""

import pytest

from pivtools_core.paths import (
    count_vector_files,
    count_vector_files_by_name,
    list_vector_files,
)

NUM_PAIRS = 12
PADDED_FMT = "%05d.mat"
UNPADDED_FMT = "B%d.mat"


def _write(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")
    return folder


def test_lists_every_present_frame(tmp_path):
    names = [PADDED_FMT % i for i in range(1, NUM_PAIRS + 1)]
    _write(tmp_path, names)

    found = list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS)

    assert [p.name for p in found] == names


def test_returns_frame_order_when_format_is_unpadded(tmp_path):
    """``B10.mat`` sorts before ``B2.mat`` by name, so name order is not frame order."""
    names = [UNPADDED_FMT % i for i in range(1, NUM_PAIRS + 1)]
    _write(tmp_path, names)

    found = list_vector_files(tmp_path, UNPADDED_FMT, NUM_PAIRS)

    assert [p.name for p in found] == names
    assert found[0].name == "B1.mat"
    assert sorted(p.name for p in found)[0] == "B1.mat"
    # The trap: plain name sorting puts B10 second, frame order puts B2 second.
    assert sorted(p.name for p in found)[1] == "B10.mat"
    assert found[1].name == "B2.mat"


def test_first_entry_is_lowest_frame_when_run_is_partial(tmp_path):
    """A partial run starting at frame 5 must still report frame 5 first."""
    _write(tmp_path, [PADDED_FMT % i for i in (9, 5, 7)])

    found = list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS)

    assert [p.name for p in found] == [PADDED_FMT % i for i in (5, 7, 9)]


def test_ignores_non_result_mat_files(tmp_path):
    """``coordinates.mat`` and ``mask.mat`` share the extension but are not frames."""
    _write(tmp_path, [PADDED_FMT % 1, "coordinates.mat", "mask.mat", "notes.txt"])

    found = list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS)

    assert [p.name for p in found] == [PADDED_FMT % 1]


def test_ignores_frames_beyond_the_expected_count(tmp_path):
    _write(tmp_path, [PADDED_FMT % i for i in (1, NUM_PAIRS, NUM_PAIRS + 1)])

    found = list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS)

    assert [p.name for p in found] == [PADDED_FMT % 1, PADDED_FMT % NUM_PAIRS]


def test_ignores_directories_named_like_frames(tmp_path):
    (tmp_path / (PADDED_FMT % 2)).mkdir(parents=True)
    _write(tmp_path, [PADDED_FMT % 1])

    found = list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS)

    assert [p.name for p in found] == [PADDED_FMT % 1]


def test_missing_directory_is_empty_not_an_error(tmp_path):
    missing = tmp_path / "never_produced"

    assert list_vector_files(missing, PADDED_FMT, NUM_PAIRS) == []
    assert count_vector_files(missing, PADDED_FMT, NUM_PAIRS) == 0


def test_empty_directory_is_empty(tmp_path):
    assert list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS) == []
    assert count_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS) == 0


def test_count_matches_list_length(tmp_path):
    _write(tmp_path, [PADDED_FMT % i for i in (1, 3, 4, 8)] + ["coordinates.mat"])

    assert count_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS) == 4
    assert count_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS) == len(
        list_vector_files(tmp_path, PADDED_FMT, NUM_PAIRS)
    )


def test_count_by_name_takes_a_prebuilt_set(tmp_path):
    """The multi-dataset scan builds the expected set once and reuses it."""
    _write(tmp_path, [PADDED_FMT % i for i in (1, 2, 5)] + ["mask.mat"])
    expected = {PADDED_FMT % i for i in range(1, NUM_PAIRS + 1)}

    assert count_vector_files_by_name(tmp_path, expected) == 3


@pytest.mark.parametrize("num_pairs", [0, 1])
def test_degenerate_pair_counts(tmp_path, num_pairs):
    _write(tmp_path, [PADDED_FMT % 1])

    found = list_vector_files(tmp_path, PADDED_FMT, num_pairs)

    assert len(found) == num_pairs
