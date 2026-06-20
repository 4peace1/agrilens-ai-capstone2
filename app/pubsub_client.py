"""
Pub/Sub publisher with a circuit breaker.

Why a circuit breaker here specifically: the publish call sits on the
critical path of `/diagnostics/{id}/complete`, which must respond fast
even on flaky infra. If Pub/Sub is degraded, we don't want every request
to hang waiting on retries — we trip the breaker, fail fast, and fall
back to an "outbox" row in Postgres that a periodic reconciler job can
re-publish once Pub/Sub recovers. This keeps the farmer-facing API
responsive even during a downstream incident (per the "Circuit Breakers
in Microservices" key concept called out in the brief).
"""
import asyncio
import time
from enum import Enum

from google.cloud import pubsub_v1

from app.config import get_settings
from app.schemas import ProcessingMessage

settings = get_settings()


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # normal operation
    OPEN = "OPEN"          # failing fast, not calling Pub/Sub
    HALF_OPEN = "HALF_OPEN"  # trial request to see if it recovered


class CircuitBreaker:
    def __init__(self, fail_threshold: int, reset_seconds: int):
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: float | None = None

    def _maybe_half_open(self) -> None:
        if (
            self.state == CircuitState.OPEN
            and self.opened_at is not None
            and time.monotonic() - self.opened_at >= self.reset_seconds
        ):
            self.state = CircuitState.HALF_OPEN

    def allow_request(self) -> bool:
        self._maybe_half_open()
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.fail_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()


_breaker = CircuitBreaker(
    fail_threshold=settings.circuit_breaker_fail_threshold,
    reset_seconds=settings.circuit_breaker_reset_seconds,
)

_publisher: pubsub_v1.PublisherClient | None = None


def _get_publisher() -> pubsub_v1.PublisherClient:
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


def _topic_path(topic: str) -> str:
    return _get_publisher().topic_path(settings.gcp_project_id, topic)


class PublishUnavailable(Exception):
    """Raised when the circuit is open; caller should fall back to the
    outbox table rather than blocking the request."""


async def publish_result_event(topic: str, payload: dict) -> None:
    """Fire-and-forget publish to the inference-results topic. Used so the
    Notification Service can fan out a push notification without the
    worker needing to know anything about FCM — keeps the worker focused
    solely on image processing + inference, per single-responsibility.
    Failures here are logged but never block/fail the worker's main job;
    a missed push notification is recoverable by the farmer polling
    GET /api/v1/diagnostics/{id}, so it is not worth tripping the
    circuit breaker over."""
    import json as _json
    import logging as _logging

    logger = _logging.getLogger("agrilens.pubsub")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _get_publisher().publish(
                _topic_path(topic), data=_json.dumps(payload).encode("utf-8")
            ),
        )
    except Exception:
        logger.warning("failed to publish result event (non-fatal)", exc_info=True)


async def publish_processing_message(message: ProcessingMessage) -> str:
    if not _breaker.allow_request():
        raise PublishUnavailable("circuit breaker open: Pub/Sub publish suspended")

    loop = asyncio.get_event_loop()
    try:
        future = await loop.run_in_executor(
            None,
            lambda: _get_publisher().publish(
                _topic_path(settings.pubsub_topic_image_processing),
                data=message.model_dump_json().encode("utf-8"),
                diagnostic_id=str(message.diagnostic_id),
                crop_type=message.crop_type.value,
                correlation_id=message.correlation_id,
            ),
        )
        message_id = await loop.run_in_executor(None, future.result, 10)
        _breaker.record_success()
        return message_id
    except Exception:
        _breaker.record_failure()
        raise
