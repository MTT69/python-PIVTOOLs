"""
Unified job manager for calibration jobs.

Provides thread-safe job tracking with timing calculations,
shared across all calibration types (scale factor, dotboard, charuco).
"""

import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional


class JobManager:
    """Thread-safe job manager for background calibration tasks."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, job_type: str, **initial_data) -> str:
        """
        Create a new job with a unique ID.

        Args:
            job_type: Type of job (e.g., 'vector', 'planar', 'charuco', 'scale_factor')
            **initial_data: Additional initial data to store with the job

        Returns:
            Unique job ID
        """
        job_id = f"{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        with self._lock:
            self._jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "start_time": time.time(),
                "error": None,
                "job_type": job_type,
                **initial_data,
            }
        return job_id

    def update_job(self, job_id: str, **kwargs) -> bool:
        """
        Update job data.

        Args:
            job_id: Job ID to update
            **kwargs: Fields to update

        Returns:
            True if job exists and was updated, False otherwise
        """
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._jobs[job_id].update(kwargs)
            return True

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job data by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            Copy of job data dict, or None if not found
        """
        with self._lock:
            if job_id not in self._jobs:
                return None
            return self._jobs[job_id].copy()

    def get_job_with_timing(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job data with added timing information.

        Adds elapsed_time and estimated_remaining based on progress.

        Args:
            job_id: Job ID to retrieve

        Returns:
            Copy of job data with timing info, or None if not found
        """
        job_data = self.get_job(job_id)
        if job_data is None:
            return None

        return self.add_timing_info(job_data)

    def add_timing_info(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add elapsed_time and estimated_remaining to job data.

        Args:
            job_data: Job data dictionary

        Returns:
            Job data with timing fields added
        """
        if "start_time" in job_data:
            elapsed = time.time() - job_data["start_time"]
            job_data["elapsed_time"] = elapsed

            progress = job_data.get("progress", 0)
            if job_data.get("status") == "running" and progress > 0:
                estimated_total = elapsed / (progress / 100.0)
                job_data["estimated_remaining"] = max(0, estimated_total - elapsed)

        return job_data

    def complete_job(self, job_id: str, **final_data) -> bool:
        """
        Mark job as completed.

        Args:
            job_id: Job ID to complete
            **final_data: Final data to store with the job

        Returns:
            True if job exists and was completed, False otherwise
        """
        return self.update_job(job_id, status="completed", progress=100, **final_data)

    def fail_job(self, job_id: str, error: str) -> bool:
        """
        Mark job as failed.

        Args:
            job_id: Job ID to fail
            error: Error message

        Returns:
            True if job exists and was marked failed, False otherwise
        """
        return self.update_job(job_id, status="failed", error=error)

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists."""
        with self._lock:
            return job_id in self._jobs

    def list_jobs(self, job_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List all jobs, optionally filtered by type.

        Args:
            job_type: Optional job type to filter by

        Returns:
            Dictionary of job_id -> job_data
        """
        with self._lock:
            if job_type is None:
                return {k: v.copy() for k, v in self._jobs.items()}
            return {
                k: v.copy()
                for k, v in self._jobs.items()
                if v.get("job_type") == job_type
            }

    def cleanup_old_jobs(self, max_age_seconds: float = 3600) -> int:
        """
        Remove jobs older than max_age_seconds that are completed or failed.

        Args:
            max_age_seconds: Maximum age in seconds (default 1 hour)

        Returns:
            Number of jobs removed
        """
        current_time = time.time()
        removed = 0

        with self._lock:
            to_remove = []
            for job_id, job_data in self._jobs.items():
                if job_data.get("status") in ("completed", "failed"):
                    start_time = job_data.get("start_time", current_time)
                    if current_time - start_time > max_age_seconds:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]
                removed += 1

        return removed


# Global instance for use across views
job_manager = JobManager()
