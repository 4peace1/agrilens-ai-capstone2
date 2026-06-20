"""
GCS storage helper.

Per the brief's guidance: "Use signed URLs for GCS uploads to keep the
API gateway lightweight" and "Don't store raw images in the PostgreSQL
database; use GCS and store the URI." The gateway never touches image
bytes — it only mints a short-lived signed PUT URL and records the
resulting `gs://` URI.
"""
from datetime import timedelta
from uuid import UUID

from google.cloud import storage

from app.config import get_settings

settings = get_settings()

_client: storage.Client | None = None


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        # When STORAGE_EMULATOR_HOST is set (local/dev/docker-compose),
        # the client library automatically talks to the emulator instead
        # of real GCS — no code branching needed.
        _client = storage.Client(project=settings.gcp_project_id)
    return _client


def raw_object_path(diagnostic_id: UUID) -> str:
    return f"raw/{diagnostic_id}.bin"


def processed_object_path(diagnostic_id: UUID) -> str:
    return f"processed/{diagnostic_id}.jpg"


def generate_upload_url(diagnostic_id: UUID, content_type: str) -> tuple[str, str]:
    """Returns (signed_put_url, gcs_uri) for the raw image bucket."""
    client = _get_client()
    bucket = client.bucket(settings.gcs_bucket_raw)
    blob = bucket.blob(raw_object_path(diagnostic_id))

    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=settings.signed_url_expiration_minutes),
        method="PUT",
        content_type=content_type,
    )
    gcs_uri = f"gs://{settings.gcs_bucket_raw}/{raw_object_path(diagnostic_id)}"
    return url, gcs_uri


def object_exists(gcs_uri: str) -> bool:
    """Used by /complete to verify the upload actually landed before we
    queue downstream processing — avoids wasting a worker invocation on
    a client that called /complete without finishing the PUT."""
    client = _get_client()
    bucket_name, _, object_path = gcs_uri.removeprefix("gs://").partition("/")
    blob = client.bucket(bucket_name).blob(object_path)
    return blob.exists()
