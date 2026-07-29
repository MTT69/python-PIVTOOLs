"""Concurrency tests for :meth:`pivtools_core.config.Config.save`.

The Flask backend runs with ``threaded=True``, so several ``/backend/update_config``
requests can sit inside ``Config.save()` at once. A save must therefore stage its
YAML in a temp file whose name is unique per writer -- a shared fixed name makes
the first ``os.replace`` consume the file out from under the second, which then
fails with ``FileNotFoundError``.
"""

import shutil
import threading
from pathlib import Path

import pytest
import yaml

from pivtools_core.config import Config

CONCURRENT_WRITERS = 8
SAVES_PER_WRITER = 40


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A writable copy of the test config.yaml, isolated per test."""
    source = Path(__file__).parent / "config.yaml"
    destination = tmp_path / "config.yaml"
    shutil.copy2(source, destination)
    return destination


def _hammer_save(cfg: Config, barrier: threading.Barrier, errors: list) -> None:
    """Save repeatedly, starting in lockstep with the other writers."""
    barrier.wait()
    for _ in range(SAVES_PER_WRITER):
        try:
            cfg.save()
        except Exception as exc:  # noqa: BLE001 - the test is what may surface
            errors.append(exc)


def test_concurrent_saves_do_not_race_on_the_temp_file(config_copy: Path) -> None:
    """Parallel saves all succeed and leave a readable config behind."""
    cfg = Config(config_copy)
    errors: list = []
    barrier = threading.Barrier(CONCURRENT_WRITERS)

    threads = [
        threading.Thread(target=_hammer_save, args=(cfg, barrier, errors))
        for _ in range(CONCURRENT_WRITERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent Config.save() raised: {errors[:3]}"
    assert config_copy.exists()
    with open(config_copy, "r", encoding="utf-8") as f:
        assert yaml.safe_load(f) is not None


def test_concurrent_saves_leave_no_temp_files(config_copy: Path) -> None:
    """Every staged temp file is consumed by its own os.replace."""
    cfg = Config(config_copy)
    errors: list = []
    barrier = threading.Barrier(CONCURRENT_WRITERS)

    threads = [
        threading.Thread(target=_hammer_save, args=(cfg, barrier, errors))
        for _ in range(CONCURRENT_WRITERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    leftovers = list(config_copy.parent.glob("*.tmp"))
    assert not leftovers, f"orphaned temp files: {[p.name for p in leftovers]}"


def test_save_cleans_up_temp_file_when_write_fails(
    config_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed dump must not litter the config directory with temp files."""
    cfg = Config(config_copy)

    def exploding_dump(*args, **kwargs):
        raise RuntimeError("simulated dump failure")

    monkeypatch.setattr("pivtools_core.config.yaml.dump", exploding_dump)

    with pytest.raises(RuntimeError, match="simulated dump failure"):
        cfg.save()

    leftovers = list(config_copy.parent.glob("*.tmp"))
    assert not leftovers, f"orphaned temp files: {[p.name for p in leftovers]}"
