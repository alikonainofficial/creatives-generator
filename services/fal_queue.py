from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from logging_config import get_logger


@dataclass
class QueueSubmitResult:
    request_id: str
    status_url: str
    response_url: str


class FalQueueClient:
    """Thin wrapper around FAL's queue HTTP API for long-running models.

    Uses the raw HTTP queue endpoints (same as the n8n pipeline) rather than
    fal_client.subscribe(), which blocks — this lets us poll incrementally and
    report progress to callers.
    """

    BASE_URL = "https://queue.fal.run"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("FAL_KEY", "")
        self._headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
        self._logger = get_logger(__name__, service="fal_queue")

    def submit(self, model: str, arguments: dict[str, Any]) -> QueueSubmitResult:
        """Submit a job to the FAL queue and return status/response URLs."""
        url = f"{self.BASE_URL}/{model}"
        start = time.time()
        with httpx.Client(timeout=60.0, transport=httpx.HTTPTransport(retries=3)) as client:
            resp = client.post(url, headers=self._headers, json=arguments)
            resp.raise_for_status()
            data = resp.json()

        elapsed = time.time() - start
        self._logger.info(
            "FAL queue job submitted",
            extra={
                "model": model,
                "request_id": data.get("request_id"),
                "duration_seconds": round(elapsed, 2),
            },
        )
        return QueueSubmitResult(
            request_id=data["request_id"],
            status_url=data["status_url"],
            response_url=data["response_url"],
        )

    def poll_status(self, status_url: str) -> dict[str, Any]:
        """Check the current status of a queued job."""
        with httpx.Client(timeout=30.0, transport=httpx.HTTPTransport(retries=3)) as client:
            resp = client.get(status_url, headers=self._headers)
            resp.raise_for_status()
        return resp.json()

    def get_result(self, response_url: str) -> dict[str, Any]:
        """Fetch the completed result from the response URL."""
        with httpx.Client(timeout=60.0, transport=httpx.HTTPTransport(retries=3)) as client:
            resp = client.get(response_url, headers=self._headers)
            resp.raise_for_status()
        return resp.json()

    def wait_for_completion(
        self,
        submit_result: QueueSubmitResult,
        poll_interval: int = 30,
        max_attempts: int = 40,
        on_poll: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Block-poll until the job reaches COMPLETED status.

        Raises TimeoutError if max_attempts is exceeded, RuntimeError on FAILED/CANCELLED.
        poll_interval=30, max_attempts=40 → up to 20 min, matching n8n's 10×120s behaviour
        at finer granularity.
        """
        for attempt in range(1, max_attempts + 1):
            time.sleep(poll_interval)
            status_data = self.poll_status(submit_result.status_url)
            status = status_data.get("status", "")

            self._logger.info(
                "FAL queue poll",
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "status": status,
                    "request_id": submit_result.request_id,
                },
            )

            if on_poll:
                on_poll(attempt, status)

            if status == "COMPLETED":
                return self.get_result(submit_result.response_url)

            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"FAL job {submit_result.request_id} ended with status '{status}'. "
                    f"Detail: {status_data}"
                )

            # IN_QUEUE or IN_PROGRESS — keep polling

        raise TimeoutError(
            f"FAL job {submit_result.request_id} did not complete after "
            f"{max_attempts} × {poll_interval}s polls ({max_attempts * poll_interval}s total)."
        )
