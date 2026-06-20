"""
Unit tests covering the pieces of the system that are pure logic and
don't require live GCP infra — these are exactly the tests a CI pipeline
should run on every PR. Integration tests against the Pub/Sub/GCS
emulators (see docker-compose.yml) would layer on top of these in a
`tests/integration/` directory, omitted here for brevity.
"""
import pytest

from app.db import validate_coordinates
from app.privacy import mask_coordinates
from app.pubsub_client import CircuitBreaker, CircuitState
from inference.model_server.ab_router import ROUTING_TABLE, select_variant


class TestCoordinateValidation:
    def test_valid_coordinates_pass(self):
        validate_coordinates(6.5244, 3.3792)  # Lagos, Nigeria

    @pytest.mark.parametrize("lat,lon", [(91, 0), (-91, 0), (0, 181), (0, -181)])
    def test_out_of_range_rejected(self, lat, lon):
        with pytest.raises(ValueError):
            validate_coordinates(lat, lon)

    def test_null_island_rejected(self):
        with pytest.raises(ValueError):
            validate_coordinates(0.0, 0.0)


class TestPrivacyMasking:
    def test_masking_is_deterministic(self):
        a = mask_coordinates(6.5244, 3.3792, diagnostic_id="abc-123")
        b = mask_coordinates(6.5244, 3.3792, diagnostic_id="abc-123")
        assert a == b

    def test_masking_differs_by_diagnostic_id(self):
        a = mask_coordinates(6.5244, 3.3792, diagnostic_id="abc-123")
        b = mask_coordinates(6.5244, 3.3792, diagnostic_id="xyz-789")
        assert a != b

    def test_masked_point_stays_close_to_original(self):
        lat, lon = 6.5244, 3.3792
        masked_lat, masked_lon = mask_coordinates(lat, lon, diagnostic_id="abc-123")
        # Grid snap (~1.1km) + jitter (250m) should never drift more than
        # ~0.03 degrees (~3.3km) from the true point.
        assert abs(masked_lat - lat) < 0.03
        assert abs(masked_lon - lon) < 0.03


class TestModelABRouter:
    def test_routing_is_deterministic_per_key(self):
        v1 = select_variant("cassava", routing_key="diagnostic-001")
        v2 = select_variant("cassava", routing_key="diagnostic-001")
        assert v1 == v2

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError):
            select_variant("durian", routing_key="diagnostic-001")

    def test_weights_sum_to_one_for_every_crop(self):
        for crop, variants in ROUTING_TABLE.items():
            total = sum(v.weight for v in variants)
            assert abs(total - 1.0) < 1e-6, f"{crop} weights must sum to 1.0"

    def test_routing_roughly_matches_configured_weights(self):
        # Statistical sanity check across many distinct keys.
        counts = {"cassava-v3-stable": 0, "cassava-v4-canary": 0}
        n = 2000
        for i in range(n):
            counts[select_variant("cassava", routing_key=f"key-{i}")] += 1
        canary_ratio = counts["cassava-v4-canary"] / n
        assert 0.05 < canary_ratio < 0.15  # configured weight is 0.10


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(fail_threshold=3, reset_seconds=10)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(fail_threshold=3, reset_seconds=10)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(fail_threshold=3, reset_seconds=10)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED
