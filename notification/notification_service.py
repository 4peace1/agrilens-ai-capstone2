"""
Notification Service.

Subscribes to the `inference-results-topic` (published by the worker on
COMPLETED/FAILED) and pushes a notification to the farmer's device via
Firebase Cloud Messaging. Deliberately decoupled from the worker so that:
  - a notification-delivery outage (FCM down) never blocks or slows
    image processing, and
  - this is the natural place to add future fan-out (SMS gateway for
    farmers without smartphones, WhatsApp Business API, etc.) without
    touching the processing pipeline at all.

This mirrors the "Notification Service" / "Push Result" node in the
brief's architecture diagram.
"""
import json
import logging

import requests
from google.cloud import pubsub_v1

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrilens.notifications")
settings = get_settings()

FCM_SEND_URL = "https://fcm.googleapis.com/fcm/send"

NOTIFICATION_COPY = {
    "COMPLETED": "Your crop diagnosis is ready — tap to view the result.",
    "FAILED": "We couldn't process your last photo. Please try again.",
}


def send_push(device_token: str, status: str, diagnostic_id: str) -> None:
    body = NOTIFICATION_COPY.get(status, "Your diagnosis status has updated.")
    response = requests.post(
        FCM_SEND_URL,
        headers={
            "Authorization": f"key={settings.fcm_server_key}",
            "Content-Type": "application/json",
        },
        json={
            "to": device_token,
            "notification": {"title": "AgriLens AI", "body": body},
            "data": {"diagnostic_id": diagnostic_id, "status": status},
        },
        timeout=5.0,
    )
    if response.status_code != 200:
        logger.warning(
            "FCM push failed for diagnostic_id=%s status_code=%s body=%s",
            diagnostic_id, response.status_code, response.text,
        )


async def _lookup_device_token(diagnostic_id: str) -> str | None:
    """Resolve the diagnostic's farmer_device_id to a registered FCM
    token. In production this hits a small device-registry table; kept
    as a stub here since device-token registration is outside this
    capstone's scope."""
    from app.db import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT farmer_device_id FROM diagnostics WHERE diagnostic_id = $1",
            diagnostic_id,
        )
    return row["farmer_device_id"] if row else None


def _callback(message) -> None:
    import asyncio

    try:
        payload = json.loads(message.data.decode("utf-8"))
        diagnostic_id = payload["diagnostic_id"]
        status = payload["status"]
        device_token = asyncio.run(_lookup_device_token(diagnostic_id))
        if device_token:
            send_push(device_token, status, diagnostic_id)
        message.ack()
    except Exception:
        logger.exception("failed to process notification event — nacking")
        message.nack()


def main() -> None:
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        settings.gcp_project_id, "inference-results-sub"
    )
    future = subscriber.subscribe(subscription_path, callback=_callback)
    logger.info("notification service listening on %s", subscription_path)
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()


if __name__ == "__main__":
    main()
