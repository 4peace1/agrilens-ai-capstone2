"""
Geospatial Data Integrity & Privacy Shield (Phase 3).

Smallholder farm coordinates are sensitive: precise GPS points can reveal
land ownership, household location, and crop value to third parties. This
module enforces a strict boundary — exact coordinates are stored once
(for legitimate internal use: spatial clustering of outbreaks, agronomist
dispatch) but are NEVER returned verbatim across the API boundary.

Two layers of protection are applied to anything leaving the service:
  1. Grid snapping — round to a coarse grid (~1km by default) so the
     exact point can't be reconstructed from repeated queries.
  2. Deterministic jitter — a small, per-device-id-seeded offset so the
     same diagnostic always masks to the same point (stable for the UI)
     without ever exposing the true coordinate.
"""
import hashlib
import math

from app.config import get_settings

settings = get_settings()


def _deterministic_jitter(seed: str, max_meters: float) -> tuple[float, float]:
    """Derive a stable, reproducible (dx, dy) offset in meters from a seed.

    Using a hash instead of `random` keeps masking deterministic per
    diagnostic — the same record always returns the same masked point —
    which matters for caching and for not leaking the true location via
    statistical averaging across repeated requests.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    angle = (int(digest[:8], 16) % 3600) / 10.0  # 0-359.9 degrees
    magnitude = (int(digest[8:16], 16) % 1000) / 1000.0 * max_meters
    dx = magnitude * math.cos(math.radians(angle))
    dy = magnitude * math.sin(math.radians(angle))
    return dx, dy


def mask_coordinates(
    lat: float, lon: float, *, diagnostic_id: str
) -> tuple[float, float]:
    """Snap-to-grid + deterministic jitter. Never exposes the raw point."""
    precision = settings.geo_mask_precision_degrees
    snapped_lat = round(lat / precision) * precision
    snapped_lon = round(lon / precision) * precision

    dx, dy = _deterministic_jitter(diagnostic_id, settings.geo_jitter_meters)
    # Convert meter offsets to degrees (approximate, fine at this precision).
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(snapped_lat))
    masked_lat = snapped_lat + (dy / meters_per_deg_lat)
    masked_lon = snapped_lon + (
        dx / meters_per_deg_lon if meters_per_deg_lon != 0 else 0
    )
    return round(masked_lat, 5), round(masked_lon, 5)
