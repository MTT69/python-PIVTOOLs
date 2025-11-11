"""
PIV Runner: Subprocess-based execution for full computational performance.

This module allows Flask to spawn PIV computations as separate subprocesses,
avoiding GIL limitations and keeping the server responsive while maintaining
full access to computational resources.
"""
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class PIVProcess:
    """Manages a single PIV computation subprocess."""

    def __init__(self, process: subprocess.Popen, job_id: str, log_file: Path):
        self.process = process
        self.job_id = job_id
        self.log_file = log_file
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.return_code: Optional[int] = None
        self._monitor_thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        """Check if the process is still running."""
        if self.process.poll() is None:
            return True
        # Process has terminated
        if self.return_code is None:
            self.return_code = self.process.returncode
            self.end_time = datetime.now()
        return False

    def cancel(self) -> bool:
        """Attempt to terminate the PIV process."""
        if self.is_running():
            try:
                self.process.terminate()
                # Give it 5 seconds to terminate gracefully
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self.process.kill()
                    self.process.wait()
                self.return_code = self.process.returncode
                self.end_time = datetime.now()
                logger.info(f"PIV job {self.job_id} cancelled")
                return True
            except Exception as e:
                logger.error(f"Error cancelling PIV job {self.job_id}: {e}")
                return False
        return False

    def get_status(self) -> dict:
        """Get current status information."""
        is_running = self.is_running()
        elapsed = (
            (self.end_time or datetime.now()) - self.start_time
        ).total_seconds()

        # Try to read recent log lines
        log_tail = []
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    log_tail = f.readlines()[-20:]  # Last 20 lines
            except Exception as e:
                logger.warning(f"Could not read log file: {e}")

        return {
            "job_id": self.job_id,
            "running": is_running,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": elapsed,
            "return_code": self.return_code,
            "log_file": str(self.log_file),
            "log_tail": log_tail,
        }


class PIVRunner:
    """Manages PIV subprocess execution and tracking."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.log_dir = project_root / "logs" / "piv_runs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_jobs: dict[str, PIVProcess] = {}
        self._lock = threading.Lock()

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"piv_{timestamp}"

    def _get_python_executable(self) -> str:
        """Get the Python executable from the virtual environment."""
        import platform
        
        # Check for virtual environment
        if platform.system() == "Windows":
            venv_python = self.project_root / "env" / "Scripts" / "python.exe"
        else:
            venv_python = self.project_root / "env" / "bin" / "python"
        
        if venv_python.exists():
            return str(venv_python)
        
        # Also check for "piv" environment (legacy)
        if platform.system() == "Windows":
            venv_python = self.project_root / "piv" / "Scripts" / "python.exe"
        else:
            venv_python = self.project_root / "piv" / "bin" / "python"
        
        if venv_python.exists():
            return str(venv_python)
        
        # Fallback to current Python
        return sys.executable

    def start_piv_job(
        self,
        cameras: Optional[list[int]] = None,
        source_path_idx: int = 0,
        base_path_idx: int = 0,
        config_overrides: Optional[dict] = None,
    ) -> dict:
        """
        Start a new PIV computation job as a subprocess.
        
        Note: Currently runs example.py which reads all settings from config.yaml.
        The parameters are accepted for API compatibility but not yet used.
        Future enhancement: Pass parameters via CLI args or environment variables.

        Parameters
        ----------
        cameras : list[int], optional
            List of camera numbers to process (future feature).
        source_path_idx : int
            Index of source path to use from config (future feature).
        base_path_idx : int
            Index of base path to use from config (future feature).
        config_overrides : dict, optional
            Configuration overrides to apply before running (future feature).

        Returns
        -------
        dict
            Job information including job_id and status.
        """
        job_id = self._generate_job_id()
        log_file = self.log_dir / f"{job_id}.log"

        # Build command - runs example.py which uses config.yaml
        python_exe = self._get_python_executable()
        script_path = self.project_root / "pivtools_core" / "example.py"
        cmd = [python_exe, str(script_path)]

        # Open log file
        log_handle = open(log_file, "w", buffering=1)  # Line buffered

        try:
            # Start subprocess
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(self.project_root),
                env=None,  # Inherit environment
            )

            piv_process = PIVProcess(process, job_id, log_file)

            with self._lock:
                self.active_jobs[job_id] = piv_process

            logger.info(f"Started PIV job {job_id} with PID {process.pid}")

            # Start a monitoring thread to clean up when done
            def monitor():
                process.wait()
                log_handle.close()
                logger.info(
                    f"PIV job {job_id} completed with return code {process.returncode}"
                )

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            piv_process._monitor_thread = monitor_thread

            return {
                "status": "started",
                "job_id": job_id,
                "pid": process.pid,
                "log_file": str(log_file),
            }

        except Exception as e:
            log_handle.close()
            logger.error(f"Failed to start PIV job: {e}")
            return {"status": "error", "message": str(e)}

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get status of a specific job."""
        with self._lock:
            piv_process = self.active_jobs.get(job_id)
            if piv_process:
                return piv_process.get_status()
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        with self._lock:
            piv_process = self.active_jobs.get(job_id)
            if piv_process:
                return piv_process.cancel()
        return False

    def list_jobs(self) -> list[dict]:
        """List all tracked jobs."""
        with self._lock:
            return [p.get_status() for p in self.active_jobs.values()]

    def cleanup_finished_jobs(self, keep_recent: int = 10):
        """Remove finished jobs from tracking, keeping only recent ones."""
        with self._lock:
            finished = [
                (jid, p)
                for jid, p in self.active_jobs.items()
                if not p.is_running()
            ]
            # Sort by end time
            finished.sort(key=lambda x: x[1].end_time or datetime.min, reverse=True)
            # Remove all but the most recent
            for jid, _ in finished[keep_recent:]:
                del self.active_jobs[jid]


# Global runner instance
_runner: Optional[PIVRunner] = None


def get_runner(project_root: Optional[Path] = None) -> PIVRunner:
    """Get or create the global PIV runner instance."""
    global _runner
    if _runner is None:
        if project_root is None:
            # Try to infer from current file location
            project_root = Path(__file__).parent.parent
        _runner = PIVRunner(project_root)
    return _runner
