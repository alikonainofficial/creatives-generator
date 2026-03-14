"""
Centralized parallel batch execution engine for Streamlit pipelines.

Pattern: daemon thread + ThreadPoolExecutor + queue.Queue → Streamlit event polling.
Workers run in a daemon-backed thread pool; the Streamlit render loop drains the
event queue on each st.rerun() cycle until EVT_ALL_DONE arrives.

Design notes (2026 best practices):
- daemon=True so the thread never prevents clean process exit.
- ThreadPoolExecutor context manager guarantees all futures are awaited and
  resources released even when exceptions occur.
- queue.Queue is the only shared state between worker threads and the render
  thread — no locks needed for the queue itself (it's thread-safe by design).
- Each worker is responsible for putting its own start / progress / result
  events on the queue; the runner only adds the final EVT_ALL_DONE sentinel.
- Inter-worker shared state (e.g. a video cache dict) must be protected by
  a threading.Lock supplied by the caller.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

# ── Worker-count bounds used across all batch modes ───────────────────────────
DEFAULT_BATCH_WORKERS = 3
MAX_BATCH_WORKERS = 10

# ── Standard event types ──────────────────────────────────────────────────────
# Workers may emit additional pipeline-specific event types beyond these.
EVT_JOB_START = "job_start"
EVT_JOB_PROGRESS = "job_progress"  # data: {"step": str, "msg": str}
EVT_JOB_RESULT = "_result"         # data: BatchWorkerResult
EVT_ALL_DONE = "_done"             # sentinel: all workers have finished


@dataclass
class BatchWorkerResult:
    """Payload emitted by each worker as an EVT_JOB_RESULT event."""

    job_key: str
    success: bool
    result: Any = None
    error: str | None = None


class ParallelBatchRunner:
    """
    Parallel batch runner for Streamlit-based pipelines.

    Launches a daemon thread that owns a ThreadPoolExecutor, keeping the
    Streamlit render thread unblocked while jobs execute concurrently.

    Worker function contract
    ------------------------
    ``worker_fn(job: Any) -> Any``

    Each worker is fully responsible for:
    - Putting ``(job_key, EVT_JOB_START, None)`` at the start.
    - Putting ``(job_key, EVT_JOB_PROGRESS, {"step": str, "msg": str})`` for
      progress updates.
    - Putting ``(job_key, EVT_JOB_RESULT, BatchWorkerResult(...))`` when done.
    - Catching all exceptions internally and expressing failures via
      ``BatchWorkerResult(success=False, error=...)``.

    The runner posts ``("__all__", EVT_ALL_DONE, None)`` once after every
    worker has finished, regardless of individual success/failure.

    Usage
    -----
    ::

        event_queue: queue.Queue = queue.Queue()

        def my_worker(job):
            event_queue.put((job.key, EVT_JOB_START, None))
            try:
                result = run_heavy_task(job)
                event_queue.put((job.key, EVT_JOB_RESULT,
                                  BatchWorkerResult(job.key, True, result)))
            except Exception as exc:
                event_queue.put((job.key, EVT_JOB_RESULT,
                                  BatchWorkerResult(job.key, False, error=str(exc))))

        runner = ParallelBatchRunner(jobs, my_worker, num_workers=3, event_queue=event_queue)
        st.session_state["eq"] = event_queue
        runner.start()   # non-blocking; returns immediately
    """

    def __init__(
        self,
        jobs: list,
        worker_fn: Callable[[Any], Any],
        num_workers: int,
        event_queue: queue.Queue | None = None,
    ) -> None:
        if not jobs:
            raise ValueError("jobs must be non-empty")
        self.jobs = jobs
        self.worker_fn = worker_fn
        self.actual_workers = max(1, min(num_workers, len(jobs), MAX_BATCH_WORKERS))
        # Accept an externally created queue so callers can store it in
        # st.session_state before the runner is started.
        self.event_queue: queue.Queue[tuple[str, str, Any]] = (
            event_queue if event_queue is not None else queue.Queue()
        )

    def start(self) -> None:
        """Start the background executor thread (returns immediately)."""
        # Capture locals so the thread closure doesn't hold a reference to
        # the entire runner object (prevents accidental GC issues).
        event_queue = self.event_queue
        worker_fn = self.worker_fn
        jobs = self.jobs
        actual_workers = self.actual_workers

        def _executor_thread() -> None:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {executor.submit(worker_fn, job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        # Worker functions should catch their own exceptions and
                        # emit a failure BatchWorkerResult. This guard handles
                        # unexpected crashes that bypass the worker's try/except.
                        pass
            event_queue.put(("__all__", EVT_ALL_DONE, None))

        thread = threading.Thread(target=_executor_thread, daemon=True)
        thread.start()
