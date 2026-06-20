"""
Outbox reconciler — run as a Kubernetes CronJob every minute.

Replays any `pubsub_outbox` rows that were written while the gateway's
circuit breaker was OPEN (Pub/Sub was unavailable). This closes the loop
on the resilience pattern in app/pubsub_client.py + app/outbox.py: the
farmer-facing API never blocked, and now that Pub/Sub has recovered, the
queued processing messages get published.
"""
import asyncio
import json
import logging

from app import db, outbox
from app.pubsub_client import publish_processing_message
from app.schemas import ProcessingMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrilens.reconciler")


async def run_once() -> int:
    await db.init_pool()
    rows = await outbox.fetch_unpublished()
    published = 0
    for row in rows:
        try:
            message = ProcessingMessage(**json.loads(row["payload"]))
            await publish_processing_message(message)
            await db.mark_queued(message.diagnostic_id)
            await outbox.mark_published(row["id"])
            published += 1
        except Exception:
            logger.exception("reconcile failed for outbox row id=%s", row["id"])
    logger.info("reconciled %s/%s outbox rows", published, len(rows))
    return published


if __name__ == "__main__":
    asyncio.run(run_once())
