"""
Gateway endpoint tests with the GCP/DB boundary mocked out — verifies the
HTTP contract and status-machine transitions without needing live infra.
Run the docker-compose stack for true integration coverage.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import DiagnosticStatus


@pytest.fixture
def client():
    with patch("app.main.db.init_pool", new_callable=AsyncMock), patch(
        "app.main.outbox.ensure_table", new_callable=AsyncMock
    ):
        with TestClient(app) as c:
            yield c


def _fake_record(status="PENDING", **overrides):
    base = {
        "diagnostic_id": uuid.uuid4(),
        "crop_type": "cassava",
        "status": status,
        "captured_at": datetime.now(timezone.utc),
        "lat": 6.5244,
        "lon": 3.3792,
        "predicted_class": None,
        "confidence": None,
        "model_version": None,
        "inference_latency_ms": None,
        "error_message": None,
    }
    base.update(overrides)
    return base


class TestCreateDiagnostic:
    @patch("app.main.db.create_diagnostic", new_callable=AsyncMock)
    @patch("app.main.storage.generate_upload_url")
    def test_returns_signed_url_and_pending_status(self, mock_url, mock_create, client):
        mock_url.return_value = ("https://signed.example/put", "gs://bucket/raw/x.bin")

        response = client.post(
            "/api/v1/diagnostics",
            json={
                "crop_type": "cassava",
                "latitude": 6.5244,
                "longitude": 3.3792,
                "captured_at": "2026-06-19T10:00:00Z",
                "image_content_type": "image/jpeg",
                "farmer_device_id": "device-abc-123",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == DiagnosticStatus.PENDING.value
        assert body["upload_url"] == "https://signed.example/put"
        mock_create.assert_awaited_once()

    def test_rejects_invalid_latitude(self, client):
        response = client.post(
            "/api/v1/diagnostics",
            json={
                "crop_type": "cassava",
                "latitude": 999,
                "longitude": 3.3792,
                "captured_at": "2026-06-19T10:00:00Z",
                "farmer_device_id": "device-abc-123",
            },
        )
        assert response.status_code == 422


class TestGetDiagnosticStatus:
    @patch("app.main.db.get_diagnostic", new_callable=AsyncMock)
    def test_not_found_returns_404(self, mock_get, client):
        mock_get.return_value = None
        response = client.get(f"/api/v1/diagnostics/{uuid.uuid4()}")
        assert response.status_code == 404

    @patch("app.main.db.get_diagnostic", new_callable=AsyncMock)
    def test_location_is_masked_not_exact(self, mock_get, client):
        record = _fake_record()
        mock_get.return_value = record

        response = client.get(f"/api/v1/diagnostics/{record['diagnostic_id']}")
        assert response.status_code == 200
        body = response.json()
        # The masked point must never exactly equal the raw stored point.
        assert (body["masked_latitude"], body["masked_longitude"]) != (
            record["lat"],
            record["lon"],
        )

    @patch("app.main.db.get_diagnostic", new_callable=AsyncMock)
    def test_completed_includes_result(self, mock_get, client):
        record = _fake_record(
            status="COMPLETED",
            predicted_class="mosaic_disease",
            confidence=0.93,
            model_version="cassava-v3-stable",
            inference_latency_ms=412.0,
        )
        mock_get.return_value = record

        response = client.get(f"/api/v1/diagnostics/{record['diagnostic_id']}")
        body = response.json()
        assert body["result"]["predicted_class"] == "mosaic_disease"
        assert body["result"]["confidence"] == 0.93
