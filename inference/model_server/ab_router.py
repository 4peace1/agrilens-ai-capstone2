"""
Model A/B Router.

Lets us run a canary model version (e.g. a newly-quantized cassava model)
against a small slice of live traffic before fully cutting over, without
any change to the gateway or worker. Routing is deterministic per
diagnostic_id (consistent hashing) so repeated/retried inference calls
for the *same* image always land on the same model version — this keeps
results reproducible if a request is retried after a timeout.
"""
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    version: str
    weight: float  # 0.0-1.0, must sum to 1.0 across variants for a crop


# Example rollout: 90% on the stable model, 10% canary on a newly
# quantized build. Swap weights to 0/100 to fully promote the canary.
ROUTING_TABLE: dict[str, list[ModelVariant]] = {
    "cassava": [
        ModelVariant(version="cassava-v3-stable", weight=0.9),
        ModelVariant(version="cassava-v4-canary", weight=0.1),
    ],
    "cocoa": [
        ModelVariant(version="cocoa-v2-stable", weight=1.0),
    ],
    "maize": [
        ModelVariant(version="maize-v1-stable", weight=1.0),
    ],
}


def select_variant(crop_type: str, routing_key: str) -> str:
    """`routing_key` is typically the diagnostic_id (or image hash) so
    routing is sticky and reproducible rather than random per call."""
    variants = ROUTING_TABLE.get(crop_type)
    if not variants:
        raise ValueError(f"no model configured for crop_type={crop_type}")

    # Deterministic bucket in [0, 1) derived from the routing key.
    digest = hashlib.sha256(routing_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    cumulative = 0.0
    for variant in variants:
        cumulative += variant.weight
        if bucket < cumulative:
            return variant.version
    return variants[-1].version  # floating point safety net
