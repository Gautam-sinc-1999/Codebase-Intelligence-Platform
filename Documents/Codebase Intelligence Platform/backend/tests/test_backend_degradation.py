"""
Distinguishing a missing driver from an unreachable server (G-12).

Both produce the same visible outcome — a fallback — and they have completely different fixes:
one is a `pip install`, the other is a server to start. The original message,
"Could not connect to live Redis (No module named 'redis')", read as a connection problem and
would send someone to debug a container that was running perfectly well.
"""
import builtins

import pytest

from app.core.database import MongoDBClient, RedisClient


@pytest.fixture
def block_import(monkeypatch):
    """Makes a named package unimportable, exactly as a missing install would."""
    real_import = builtins.__import__

    def _block(name):
        def fake_import(module, *args, **kwargs):
            if module == name or module.startswith(f"{name}."):
                raise ImportError(f"No module named '{name}'")
            return real_import(module, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)

    return _block


# ------------------------------------------------------------------ the drivers are present

def test_the_declared_drivers_are_actually_installed():
    """
    requirements.txt declared motor and redis; neither was installed, so both stores silently
    degraded on every run. Nothing failed, which is why it went unnoticed.
    """
    import motor            # noqa: F401
    import redis            # noqa: F401
    import redis.asyncio    # noqa: F401


async def test_mongo_uses_the_async_driver_when_it_is_available():
    """
    With motor absent, every Mongo call goes through pymongo — a synchronous driver called from
    the event loop, which blocks it.
    """
    client = MongoDBClient()
    await client.connect()
    if client.is_fallback:
        pytest.skip("no MongoDB server reachable in this environment")
    assert client.is_pymongo is False, "fell back to the blocking synchronous driver"
    assert client.degraded_reason is None
    assert client.backend_name == "mongodb-motor", \
        f"expected the async driver, got {client.backend_name!r}"


# ------------------------------------------------------------------ the distinction

async def test_a_missing_redis_package_is_reported_as_a_missing_package(block_import, caplog):
    block_import("redis")
    client = RedisClient()
    with caplog.at_level("WARNING"):
        await client.connect()

    assert client.is_fallback is True
    assert client.unavailable_reason == "driver-not-installed"

    message = caplog.text
    assert "not installed" in message
    assert "pip install" in message, "the message must name the actual fix"
    assert "NOT a connection problem" in message, "must not read as a server being down"


async def test_an_unreachable_redis_server_is_reported_as_a_server_problem(monkeypatch, caplog):
    from app.core.config import settings
    monkeypatch.setattr(settings, "REDIS_URI", "redis://127.0.0.1:9")

    client = RedisClient()
    with caplog.at_level("WARNING"):
        await client.connect()

    assert client.is_fallback is True
    assert client.unavailable_reason == "server-unreachable"
    assert "no server is reachable" in caplog.text
    assert "pip install" not in caplog.text, "sent the reader to install a package that is present"


async def test_a_missing_motor_degrades_to_pymongo_rather_than_to_no_database(block_import, caplog):
    """Motor missing is a performance degradation, not an outage — and is therefore easy to miss."""
    block_import("motor")
    client = MongoDBClient()
    with caplog.at_level("WARNING"):
        await client.connect()

    if client.is_fallback:
        pytest.skip("no MongoDB server reachable in this environment")

    assert client.is_pymongo is True
    assert client.degraded_reason == "motor-not-installed"
    assert "block the event loop" in caplog.text, "the actual cost was not stated"


# ------------------------------------------------------------------ surfaced to the client

def test_health_reports_why_a_store_is_degraded_not_only_that_it_is(api_client, monkeypatch):
    """
    'degraded: [cache]' alone is not actionable — it is true whether the package is missing or
    the server is down.
    """
    from app.core.database import redis_client

    monkeypatch.setattr(redis_client, "is_fallback", True)
    monkeypatch.setattr(redis_client, "unavailable_reason", "driver-not-installed")

    body = api_client.get("/").json()
    assert "cache" in body["degraded"]
    assert body["degraded_reasons"]["cache"] == "driver-not-installed"


def test_health_omits_reasons_when_everything_is_healthy(api_client, monkeypatch):
    from app.core.database import redis_client, db_client

    monkeypatch.setattr(redis_client, "is_fallback", False)
    monkeypatch.setattr(redis_client, "unavailable_reason", None)
    monkeypatch.setattr(db_client, "degraded_reason", None)

    body = api_client.get("/").json()
    assert "degraded_reasons" not in body


# ------------------------------------------------------------------ the async client's lifetime

def test_the_motor_client_survives_a_change_of_event_loop():
    """
    Motor binds `AsyncIOMotorClient` to the loop that constructs it, and using it from another
    raises `RuntimeError: ... attached to a different loop`.

    A single process on a single loop never notices — which is exactly why this stayed hidden
    while the driver was not installed at all. But `db_client` is a module-level singleton and
    the loop it was bound to does not last forever: an in-process restart, a reload, or a second
    application lifespan each produce a new loop with the old client still pointing at the dead
    one. Installing motor turned that latent fragility into 19 failing tests.
    """
    import asyncio

    client = MongoDBClient()

    async def connect_here():
        await client.connect()
        return client.is_fallback, client.is_pymongo

    first_loop = asyncio.new_event_loop()
    try:
        is_fallback, is_pymongo = first_loop.run_until_complete(connect_here())
    finally:
        first_loop.close()

    if is_fallback or is_pymongo:
        pytest.skip("motor is not the active MongoDB driver in this environment")

    bound_to = client._loop
    assert bound_to is not None

    async def use_from_a_different_loop():
        # The operation that used to raise: a real query through the singleton's collection.
        await client.get_collection("repositories").find_one({"repository_id": "__no_such_repo__"})
        return client._loop

    second_loop = asyncio.new_event_loop()
    try:
        rebound_to = second_loop.run_until_complete(use_from_a_different_loop())
    finally:
        second_loop.close()

    assert rebound_to is not bound_to, "the client was not rebound to the new loop"


def test_rebinding_is_a_no_op_while_the_loop_is_unchanged():
    """Rebuilding the client on every call would throw away connection pooling for nothing."""
    from app.core.database import db_client

    if db_client.is_fallback or db_client.is_pymongo:
        pytest.skip("motor is not the active MongoDB driver in this environment")

    before = db_client.db
    db_client._rebind_if_loop_changed()
    db_client._rebind_if_loop_changed()
    assert db_client.db is before, "rebound despite the loop being unchanged"
