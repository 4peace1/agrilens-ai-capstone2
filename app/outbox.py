"""
Transactional outbox — fallback path when the Pub/Sub circuit breaker is
OPEN. Instead of blocking (or dropping) the request, we durably record
the intended message in Postgres and let a periodic reconciler
(`scripts/reconcile_outbox.py`) replay it once Pub/Sub recovers.
"""
from __future__ import annotations

from app.db import get_pool
from app.schemas import ProcessingMessage

DDL = """
CREATE TABLE IF NOT EXISTS pubsub_outbox (
    id              BIGSERIAL PRIMARY KEY,
    topic           TEXT NOT NULL,
    payload         JSONB NOT NULL,
    published       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_outbox_unpublished
    ON pubsub_outbox (published) WHERE published = FALSE;
"""


async def ensure_table() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(DDL)


async def enqueue(topic: str, message: ProcessingMessage) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO pubsub_outbox (topic, payload) VALUES ($1, $2::jsonb)",
            topic,
            message.model_dump_json(),
        )


async def fetch_unpublished(limit: int = 100):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT id, topic, payload FROM pubsub_outbox "
            "WHERE published = FALSE ORDER BY created_at LIMIT $1",
            limit,
        )


async def mark_published(row_id: int) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pubsub_outbox SET published = TRUE WHERE id = $1", row_id
        )
