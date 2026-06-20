"""
Image-processing worker (Phase 1 deliverable: "Cloud Function/Worker for
image normalization — resizing, padding, and PostGIS coordinate
validation").

Root cause this directly addresses: "unoptimized preprocessing scripts
resulting in excessive memory consumption and frequent container OOM
kills." Mitigations applied here:
  - Images are streamed from GCS in chunks rather than loaded fully into
    a request buffer first.
  - PIL's `Image.draft()` is used so large/odd-resolution budget-phone
    JPEGs are downscaled *during* decode instead of fully decoded at
    native resolution before resizing (this is the single biggest memory
    win for this workload).
  - Large intermediate arrays are deleted and `gc.collect()`'d explicitly
    after each stage rather than relying on end-of-function cleanup,
    since this process handles many images per container lifetime.
  - The consumer is idempotent (via `db.mark_processing`'s conditional
    UPDATE) so Pub/Sub's at-least-once redelivery can never double-bill
    a worker pod's memory budget on the same image twice.

Runs as a long-lived Pub/Sub *pull* subscriber (suited to GKE) — the same
handler logic (`process_message`) can be wrapped as a push-based Cloud
Function/Cloud Run endpoint with no changes, per the brief's "Cloud
Function/Worker" framing.
"""
import gc
import io
import json
import logging
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

from google.cloud import pubsub_v1, storage
from PIL import Image, ImageOps

from app import db
from app.config import get_settings
from app.schemas import ProcessingMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrilens.worker")
settings = get_settings()

TARGET_SIZE = settings.model_input_size  # square, matches model input shape
MAX_RAW_BYTES = 25 * 1024 * 1024  # reject anything absurd before decoding


def _download_streamed(gcs_uri: str) -> bytes:
    """Stream-download with a hard size cap, rather than trusting
    Content-Length blindly — protects the worker pod's memory ceiling
    against a corrupted or malicious upload."""
    client = storage.Client(project=settings.gcp_project_id)
    bucket_name, _, object_path = gcs_uri.removeprefix("gs://").partition("/")
    blob = client.bucket(bucket_name).blob(object_path)

    buf = io.BytesIO()
    with blob.open("rb", chunk_size=256 * 1024) as stream:
        while chunk := stream.read(256 * 1024):
            buf.write(chunk)
            if buf.tell() > MAX_RAW_BYTES:
                raise ValueError(f"image exceeds {MAX_RAW_BYTES} byte cap")
    return buf.getvalue()


def normalize_image(raw_bytes: bytes) -> bytes:
    """Resize + letterbox-pad to a fixed square input, handling arbitrary
    source aspect ratios from diverse budget smartphone cameras without
    distorting the subject (important for disease-spot pattern fidelity).
    """
    with Image.open(io.BytesIO(raw_bytes)) as img:
        # Decode-time downscale hint — avoids fully decoding a 12MP photo
        # just to immediately shrink it to 224x224. Big memory win.
        img.draft("RGB", (TARGET_SIZE * 2, TARGET_SIZE * 2))
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)  # respect phone camera orientation

        # Letterbox: scale to fit, then pad to square so we never distort
        # the leaf/pod shape that the classifier relies on.
        img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        offset = (
            (TARGET_SIZE - img.width) // 2,
            (TARGET_SIZE - img.height) // 2,
        )
        padded.paste(img, offset)

        out = io.BytesIO()
        padded.save(out, format="JPEG", quality=90)
        result = out.getvalue()

    # Explicit cleanup — this worker processes many images per container
    # lifetime, so we don't wait on GC's normal cadence.
    del raw_bytes
    gc.collect()
    return result


def _upload_processed(diagnostic_id: str, data: bytes) -> str:
    client = storage.Client(project=settings.gcp_project_id)
    bucket = client.bucket(settings.gcs_bucket_processed)
    object_path = f"processed/{diagnostic_id}.jpg"
    blob = bucket.blob(object_path)
    blob.upload_from_string(data, content_type="image/jpeg")
    return f"gs://{settings.gcs_bucket_processed}/{object_path}"


def process_message(message: ProcessingMessage) -> None:
    diagnostic_id = message.diagnostic_id

    # Idempotent claim — if another redelivered copy of this message already
    # claimed it, this is a silent, safe no-op.
    import asyncio

    claimed = asyncio.run(db.mark_processing(diagnostic_id))
    if not claimed:
        logger.info(
            "id=%s already PROCESSING/COMPLETED — skipping duplicate delivery",
            diagnostic_id,
        )
        return

    start = time.monotonic()
    try:
        raw_bytes = _download_streamed(message.raw_gcs_uri)
        processed_bytes = normalize_image(raw_bytes)
        processed_uri = _upload_processed(str(diagnostic_id), processed_bytes)

        # Hand off to the inference layer (Phase 2). In production this is
        # an HTTP call to the model orchestrator / A-B router; kept as a
        # thin call here to keep this module focused on preprocessing.
        from inference.client import run_inference

        result = run_inference(
            crop_type=message.crop_type, image_bytes=processed_bytes
        )

        asyncio.run(
            db.mark_completed(
                diagnostic_id,
                processed_gcs_uri=processed_uri,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                model_version=result.model_version,
                inference_latency_ms=result.latency_ms,
            )
        )
        from app.pubsub_client import publish_result_event

        asyncio.run(
            publish_result_event(
                settings.pubsub_topic_inference_results,
                {
                    "diagnostic_id": str(diagnostic_id),
                    "status": "COMPLETED",
                    "predicted_class": result.predicted_class,
                    "confidence": result.confidence,
                    "correlation_id": message.correlation_id,
                },
            )
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "id=%s corr=%s processed+inferred in %.1fms (class=%s conf=%.2f)",
            diagnostic_id,
            message.correlation_id,
            elapsed_ms,
            result.predicted_class,
            result.confidence,
        )
    except Exception as exc:  # noqa: BLE001 — must not crash the subscriber
        logger.exception("processing failed for id=%s", diagnostic_id)
        asyncio.run(db.mark_failed(diagnostic_id, str(exc)))
        from app.pubsub_client import publish_result_event

        asyncio.run(
            publish_result_event(
                settings.pubsub_topic_inference_results,
                {
                    "diagnostic_id": str(diagnostic_id),
                    "status": "FAILED",
                    "correlation_id": message.correlation_id,
                },
            )
        )


def _callback(pubsub_message) -> None:
    try:
        payload = json.loads(pubsub_message.data.decode("utf-8"))
        message = ProcessingMessage(**payload)
        process_message(message)
        pubsub_message.ack()
    except Exception:
        logger.exception("unrecoverable error handling Pub/Sub message — nacking")
        pubsub_message.nack()


def main() -> None:
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        settings.gcp_project_id, settings.pubsub_subscription_image_processing
    )
    # flow_control caps concurrent in-flight messages, which directly
    # bounds this pod's peak memory usage — the actual fix for the OOM
    # root cause, paired with HPA scaling out more pods under load.
    flow_control = pubsub_v1.types.FlowControl(max_messages=4)

    future = subscriber.subscribe(
        subscription_path, callback=_callback, flow_control=flow_control
    )
    logger.info("worker listening on %s", subscription_path)
    try:
        future.result()
    except (KeyboardInterrupt, FutureTimeoutError):
        future.cancel()
        future.result()


if __name__ == "__main__":
    main()
