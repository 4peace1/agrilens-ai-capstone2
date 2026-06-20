"""
Centralized, environment-driven configuration.

Per the project brief: "Use environment variables for all configuration
to ensure portability between dev and prod." Nothing here is hardcoded —
every value has a sane local-dev default but is meant to be overridden
via env vars (or a k8s ConfigMap/Secret) in staging/prod.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- General ---
    environment: str = "development"
    service_name: str = "agrilens-gateway"

    # --- GCP ---
    gcp_project_id: str = "agrilens-dev"
    pubsub_topic_image_processing: str = "image-processing-topic"
    pubsub_topic_inference_results: str = "inference-results-topic"
    pubsub_subscription_image_processing: str = "image-processing-sub"
    gcs_bucket_raw: str = "agrilens-raw-images"
    gcs_bucket_processed: str = "agrilens-processed-images"
    signed_url_expiration_minutes: int = 15

    # Allows pointing the GCS/Pub/Sub clients at local emulators for
    # docker-compose / dev. In prod these are simply unset.
    pubsub_emulator_host: str | None = None
    storage_emulator_host: str | None = None

    # --- Database (PostgreSQL + PostGIS) ---
    database_url: str = (
        "postgresql://agrilens:agrilens@localhost:5432/agrilens"
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # --- Inference layer ---
    inference_service_url: str = "http://inference-orchestrator:8080"
    inference_timeout_seconds: float = 2.0  # hard SLA from the brief
    model_input_size: int = 224  # square input expected by TFLite models
    f1_score_floor: float = 0.85  # minimum acceptable model quality

    # --- Geospatial privacy shield ---
    # Coordinates returned to clients/3rd parties are snapped to this grid
    # (in decimal degrees) to protect exact farm locations. ~0.01 deg ≈ 1.1km.
    geo_mask_precision_degrees: float = 0.01
    geo_jitter_meters: float = 250.0

    # --- Notifications ---
    fcm_server_key: str = "CHANGE_ME"

    # --- Circuit breaker (Pub/Sub publish path) ---
    circuit_breaker_fail_threshold: int = 5
    circuit_breaker_reset_seconds: int = 30

    class Config:
        env_file = ".env"
        env_prefix = "AGRILENS_"
        protected_namespaces = ()


@lru_cache
def get_settings() -> Settings:
    return Settings()
