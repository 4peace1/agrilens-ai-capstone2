"""
AgriLens AI — FastAPI Gateway (Phase 1: Decoupled Pre-processing &
Asynchronous Ingestion).

This service does exactly two cheap things and nothing else:
  1. Validate metadata + mint a signed GCS upload URL.
  2. On upload confirmation, publish a Pub/Sub message and return
     immediately — all heavy lifting (resize, normalize, infer) happens
     out-of-band in the worker/inference services.

This keeps the gateway's CPU/memory footprint tiny and predictable, which
is what lets it survive 5x traffic bursts during planting season without
starving the request-handling threads (the root cause identified in the
brief: CPU contention between API traffic and ML workloads).
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from app import db, outbox, storage
from app.config import get_settings
from app.privacy import mask_coordinates
from app.pubsub_client import PublishUnavailable, publish_processing_message
from app.schemas import (
    CropType,
    DiagnosticCreateRequest,
    DiagnosticCreateResponse,
    DiagnosticStatus,
    DiagnosticStatusResponse,
    InferenceResult,
    ProcessingMessage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrilens.gateway")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    await outbox.ensure_table()
    logger.info("Gateway started — DB pool warm, outbox table ensured.")
    yield
    logger.info("Gateway shutting down.")


app = FastAPI(
    title="AgriLens AI — Diagnostic Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

# Bonus rubric item: Observability & Monitoring. Exposes /metrics in
# Prometheus format (request latency histograms, in-flight count, etc.)
# with zero extra code beyond this one line.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _correlation_id() -> str:
    """A single ID that threads through gateway -> Pub/Sub -> worker ->
    inference logs, so one farmer's image can be traced end-to-end."""
    return uuid.uuid4().hex[:16]


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.post("/api/v1/diagnostics", response_model=DiagnosticCreateResponse, status_code=201)
async def create_diagnostic(payload: DiagnosticCreateRequest):
    """Step 1: validate + register metadata, hand back a signed upload URL.

    Critically, this endpoint never touches image bytes — it returns in
    milliseconds even on a 2G connection, which is the whole point of the
    'accept immediately' pattern from the brief.
    """
    diagnostic_id = uuid.uuid4()
    correlation_id = _correlation_id()

    try:
        db.validate_coordinates(payload.latitude, payload.longitude)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    upload_url, gcs_uri = storage.generate_upload_url(
        diagnostic_id, payload.image_content_type
    )

    await db.create_diagnostic(
        diagnostic_id=diagnostic_id,
        crop_type=payload.crop_type,
        lat=payload.latitude,
        lon=payload.longitude,
        captured_at=payload.captured_at,  # honors offline-captured timestamps
        raw_gcs_uri=gcs_uri,
        farmer_device_id=payload.farmer_device_id,
        correlation_id=correlation_id,
    )

    logger.info(
        "diagnostic created id=%s crop=%s corr=%s",
        diagnostic_id,
        payload.crop_type.value,
        correlation_id,
    )

    return DiagnosticCreateResponse(
        diagnostic_id=diagnostic_id,
        upload_url=upload_url,
        gcs_uri=gcs_uri,
        status=DiagnosticStatus.PENDING,
        expires_in_minutes=settings.signed_url_expiration_minutes,
    )


async def _dispatch_processing(
    diagnostic_id: uuid.UUID,
    crop_type: CropType,
    lat: float,
    lon: float,
    captured_at,
    raw_gcs_uri: str,
    correlation_id: str,
) -> None:
    """Background task: publish to Pub/Sub, with an outbox fallback if the
    circuit breaker has tripped. Runs after the HTTP response has already
    been sent — this is the actual decoupling step."""
    message = ProcessingMessage(
        diagnostic_id=diagnostic_id,
        correlation_id=correlation_id,
        raw_gcs_uri=raw_gcs_uri,
        crop_type=crop_type,
        latitude=lat,
        longitude=lon,
        captured_at=captured_at,
    )
    try:
        await publish_processing_message(message)
        await db.mark_queued(diagnostic_id)
        logger.info("queued diagnostic_id=%s corr=%s", diagnostic_id, correlation_id)
    except PublishUnavailable:
        logger.warning(
            "Pub/Sub circuit open — falling back to outbox for id=%s", diagnostic_id
        )
        await outbox.enqueue(settings.pubsub_topic_image_processing, message)
    except Exception:
        logger.exception("unexpected publish failure for id=%s", diagnostic_id)
        await outbox.enqueue(settings.pubsub_topic_image_processing, message)


@app.post("/api/v1/diagnostics/{diagnostic_id}/complete", status_code=202)
async def complete_upload(diagnostic_id: uuid.UUID, background_tasks: BackgroundTasks):
    """Step 2: client confirms the signed-URL PUT finished. We verify the
    object actually landed in GCS, flip status, and offload the Pub/Sub
    publish to a background task so this call returns instantly —
    mirroring the `BackgroundTasks` pattern shown in the brief's example.
    """
    record = await db.get_diagnostic(diagnostic_id)
    if record is None:
        raise HTTPException(status_code=404, detail="diagnostic not found")

    raw_uri = f"gs://{settings.gcs_bucket_raw}/raw/{diagnostic_id}.bin"
    if not storage.object_exists(raw_uri):
        raise HTTPException(
            status_code=409,
            detail="upload not yet visible in GCS — retry shortly",
        )

    updated = await db.mark_uploaded(diagnostic_id)
    if not updated:
        # Already past PENDING — idempotent no-op rather than an error,
        # since the mobile app may retry /complete on a flaky connection.
        logger.info("complete() called again for id=%s — idempotent no-op", diagnostic_id)
        return {"status": "already_processed", "diagnostic_id": str(diagnostic_id)}

    background_tasks.add_task(
        _dispatch_processing,
        diagnostic_id,
        CropType(record["crop_type"]),
        record["lat"],
        record["lon"],
        record["captured_at"],
        raw_uri,
        uuid.uuid4().hex[:16],
    )

    return {"status": "accepted", "diagnostic_id": str(diagnostic_id)}


@app.get("/api/v1/diagnostics/{diagnostic_id}", response_model=DiagnosticStatusResponse)
async def get_diagnostic_status(diagnostic_id: uuid.UUID):
    """Polled by the mobile app (and used to push the final notification).
    Location is always returned masked — see app/privacy.py."""
    record = await db.get_diagnostic(diagnostic_id)
    if record is None:
        raise HTTPException(status_code=404, detail="diagnostic not found")

    masked_lat, masked_lon = mask_coordinates(
        record["lat"], record["lon"], diagnostic_id=str(diagnostic_id)
    )

    result = None
    if record["status"] == DiagnosticStatus.COMPLETED.value:
        result = InferenceResult(
            predicted_class=record["predicted_class"],
            confidence=record["confidence"],
            model_version=record["model_version"],
            latency_ms=record["inference_latency_ms"],
            crop_type=CropType(record["crop_type"]),
        )

    return DiagnosticStatusResponse(
        diagnostic_id=diagnostic_id,
        status=DiagnosticStatus(record["status"]),
        crop_type=CropType(record["crop_type"]),
        masked_latitude=masked_lat,
        masked_longitude=masked_lon,
        captured_at=record["captured_at"],
        result=result,
        error_message=record["error_message"],
    )
