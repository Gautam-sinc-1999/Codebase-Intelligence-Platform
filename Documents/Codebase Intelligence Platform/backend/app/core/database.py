import os
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger("database")

def sanitize_mongo_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Converts MongoDB ObjectId instances to JSON-serializable strings."""
    if not doc:
        return doc
    cleaned = dict(doc)
    if "_id" in cleaned:
        cleaned["_id"] = str(cleaned["_id"])
    return cleaned

class FallbackMongoDBStore:
    """
    JSON-persisted fallback used when no MongoDB server is reachable.

    This previously kept everything in a process dictionary despite advertising persistence, so
    every conversation, title and working context was lost on restart — and because it is the
    path taken whenever Mongo is not running, that was the default experience rather than an
    edge case. Each collection is now a JSON file under DATA_DIR/fallback_db/, loaded at startup
    and written atomically on each mutation.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join(settings.DATA_DIR, "fallback_db")
        self.collections: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()

    def _path(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}.json")

    def _load_all(self) -> None:
        if not os.path.isdir(self.storage_dir):
            return
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue
            name = filename[:-len(".json")]
            try:
                with open(self._path(name), "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.collections[name] = loaded
            except Exception as e:
                # A corrupt file must not prevent startup; the collection starts empty instead.
                logger.error("Could not load fallback collection '%s': %s", name, e)

    def save(self, name: str) -> None:
        """Writes one collection to disk atomically, so an interrupted write cannot corrupt it."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            tmp_path = f"{self._path(name)}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.collections.get(name, []), f, default=str)
            os.replace(tmp_path, self._path(name))
        except Exception as e:
            logger.error("Could not persist fallback collection '%s': %s", name, e)

    def get_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = []
        return FallbackCollection(self.collections[name], name, self)

class _UpdateResult:
    """The part of PyMongo's UpdateResult that callers of this store actually read."""

    def __init__(self, matched_count: int = 0):
        self.matched_count = matched_count
        self.modified_count = matched_count


class FallbackCollection:
    def __init__(self, data_list: List[Dict[str, Any]], name: str = "", store: Optional[FallbackMongoDBStore] = None):
        self.data = data_list
        self.name = name
        self.store = store

    async def _persist(self):
        """Flushes this collection to disk off the event loop."""
        if self.store and self.name:
            await asyncio.to_thread(self.store.save, self.name)

    async def insert_one(self, document: Dict[str, Any]):
        self.data.append(document)
        await self._persist()
        return type("InsertOneResult", (), {"inserted_id": document.get("_id", document.get("id"))})()

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        # Every key in the query has to match, as `find_one` and `delete_one` already require.
        # Matching only the first key made a two-key query silently address the wrong document:
        # `{"conversation_id": c, "seq": 3}` matched on the conversation alone and updated its
        # *first* message, so feedback recorded against one answer landed on another.
        target = None
        for item in self.data:
            if all(item.get(k) == v for k, v in query.items()):
                target = item
                break

        if not target and upsert:
            target = dict(query)
            self.data.append(target)

        if target:
            if "$set" in update:
                target.update(update["$set"])
            if "$push" in update:
                for p_key, p_val in update["$push"].items():
                    if p_key not in target or not isinstance(target[p_key], list):
                        target[p_key] = []
                    if isinstance(p_val, dict) and "$each" in p_val:
                        target[p_key].extend(p_val["$each"])
                    else:
                        target[p_key].append(p_val)
            await self._persist()

        # Reported like PyMongo's, so a caller can tell "no such document" from "updated" without
        # knowing which store it is talking to.
        return _UpdateResult(matched_count=1 if target else 0)

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for item in self.data:
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                return sanitize_mongo_doc(item)
        return None

    def find(self, query: Dict[str, Any]):
        results = []
        for item in self.data:
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                results.append(sanitize_mongo_doc(item))
        
        class Cursor:
            def __init__(self, items):
                self.items = items
            def sort(self, key, direction=-1):
                self.items.sort(key=lambda x: x.get(key, ""), reverse=(direction < 0))
                return self
            def limit(self, count):
                self.items = self.items[:count]
                return self
            async def to_list(self, length=None):
                return self.items[:length] if length else self.items
            def __aiter__(self):
                self._iter = iter(self.items)
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration
        return Cursor(results)

    async def delete_one(self, query: Dict[str, Any]):
        for i, item in enumerate(self.data):
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                del self.data[i]
                await self._persist()
                break

class PyMongoCollectionWrapper:
    """Wrapper making synchronous PyMongo collections compatible with async API & JSON serialization."""
    def __init__(self, collection):
        self.col = collection

    async def insert_one(self, document: Dict[str, Any]):
        return await asyncio.to_thread(self.col.insert_one, document)

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        return await asyncio.to_thread(self.col.update_one, query, update, upsert=upsert)

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = await asyncio.to_thread(self.col.find_one, query)
        return sanitize_mongo_doc(raw)

    def find(self, query: Dict[str, Any]):
        sync_cursor = self.col.find(query)
        
        class AsyncCursorWrapper:
            def __init__(self, cursor):
                self.cursor = cursor
            def sort(self, key, direction=-1):
                self.cursor.sort(key, direction)
                return self
            def limit(self, count):
                self.cursor.limit(count)
                return self
            async def to_list(self, length=None):
                def _fetch():
                    res = list(self.cursor)
                    cleaned = [sanitize_mongo_doc(item) for item in res]
                    return cleaned[:length] if length else cleaned
                return await asyncio.to_thread(_fetch)
        return AsyncCursorWrapper(sync_cursor)

    async def delete_one(self, query: Dict[str, Any]):
        return await asyncio.to_thread(self.col.delete_one, query)

class MongoDBClient:
    def __init__(self):
        self.db = None
        self.is_fallback = False
        self.is_pymongo = False
        self.fallback_store = FallbackMongoDBStore()
        # The event loop the motor client is bound to; see _rebind_if_loop_changed.
        self._loop = None
        # None while fully healthy. 'motor-not-installed' still means MongoDB works — through a
        # synchronous driver that blocks the event loop, which is a degradation easy to miss
        # because nothing fails. The other values mean Mongo is not being used at all.
        self.degraded_reason = None

    async def connect(self):
        # 1. Try motor (async driver)
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
            await client.server_info()
            self.db = client[settings.MONGODB_DB_NAME]
            self.is_fallback = False
            self.is_pymongo = False
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
            logger.info("Connected to MongoDB via Motor at %s (DB: %s)", settings.MONGODB_URI, settings.MONGODB_DB_NAME)
            return
        except ImportError:
            # Worth saying out loud: without motor every Mongo call is a sync driver call made
            # from the event loop, which blocks it. The pymongo path below still works, so this
            # is a performance degradation rather than an outage — and therefore easy to miss.
            self.degraded_reason = "motor-not-installed"
            logger.warning(
                "The 'motor' package is not installed, so MongoDB will be used through the "
                "synchronous pymongo driver. Calls will block the event loop. "
                "Install it with: pip install -r requirements.txt."
            )
        except Exception:
            pass

        # 2. Try pymongo (sync driver wrapper)
        try:
            import pymongo
            client = pymongo.MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
            client.server_info()
            self.db = client[settings.MONGODB_DB_NAME]
            self.is_fallback = False
            self.is_pymongo = True
            logger.info("Connected to MongoDB via PyMongo at %s (DB: %s)", settings.MONGODB_URI, settings.MONGODB_DB_NAME)
            return
        except ImportError as e:
            self.is_fallback = True
            self.degraded_reason = "driver-not-installed"
            self.db = self.fallback_store
            logger.warning(
                "Neither 'motor' nor 'pymongo' is installed (%s), so MongoDB is not being used "
                "at all — this is NOT a connection problem. Install them with: "
                "pip install -r requirements.txt. Falling back to JSON storage at %s.",
                e, os.path.abspath(self.fallback_store.storage_dir),
            )
        except Exception as e:
            self.is_fallback = True
            self.degraded_reason = "server-unreachable"
            self.db = self.fallback_store
            logger.warning(
                "No MongoDB server reachable at %s (%s). Falling back to JSON storage at %s. "
                "Conversations persist across restarts, but this is single-process only and not "
                "suitable for production.",
                settings.MONGODB_URI, e, os.path.abspath(self.fallback_store.storage_dir),
            )

    @property
    def backend_name(self) -> str:
        """Identifies the active storage backend so degraded mode is observable, not silent."""
        if self.is_fallback or self.db is None:
            return "json-file-fallback"
        return "mongodb-pymongo" if self.is_pymongo else "mongodb-motor"

    def get_collection(self, collection_name: str):
        if self.is_fallback or self.db is None:
            return self.fallback_store.get_collection(collection_name)
        if self.is_pymongo:
            return PyMongoCollectionWrapper(self.db[collection_name])
        self._rebind_if_loop_changed()
        return self.db[collection_name]

    def _rebind_if_loop_changed(self) -> None:
        """
        Rebuilds the motor client if the running event loop is not the one it was created on.

        Motor binds an `AsyncIOMotorClient` to the loop that constructs it, and using it from a
        different loop raises `RuntimeError: ... attached to a different loop`. One long-lived
        process on one loop never notices — which is why this stayed hidden while the driver was
        simply not installed — but the client here is a module-level singleton, and the loop it
        was bound to does not last forever: an in-process restart, a reload, or a second
        application lifespan all produce a new loop with the old client still pointing at the
        dead one.

        Rebuilding is cheap (construction is local; no I/O until the first operation) and is the
        only thing that makes a module-level async client safe to keep across loops.
        """
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            return  # called outside async context; nothing to rebind against

        if self._loop is current:
            return

        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
            self.db = client[settings.MONGODB_DB_NAME]
            self._loop = current
            logger.debug("Rebound the MongoDB client to a new event loop.")
        except Exception as e:
            logger.warning("Could not rebind the MongoDB client to the current loop: %s", e)

class FallbackRedisStore:
    """In-memory Redis cache fallback when redis server is not running."""
    def __init__(self):
        self.store = {}

    async def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        self.store[key] = value

    async def delete(self, key: str):
        self.store.pop(key, None)

class RedisClient:
    """Redis Manager for Active Conversation State & Short-Term Working Memory Cache."""
    def __init__(self):
        self.client = None
        self.is_fallback = False
        self.fallback_store = FallbackRedisStore()
        # None while healthy; set to 'driver-not-installed' or 'server-unreachable' on fallback,
        # because those two have completely different fixes and look identical from outside.
        self.unavailable_reason = None

    async def connect(self):
        try:
            import redis.asyncio as aioredis
            self.client = aioredis.from_url(settings.REDIS_URI, socket_timeout=1.0)
            await self.client.ping()
            self.is_fallback = False
            logger.info("Connected to Redis at %s", settings.REDIS_URI)
        except ImportError as e:
            # Distinguished from an unreachable server deliberately. "Could not connect to live
            # Redis (No module named 'redis')" reads as a server problem and sends someone to
            # check a container that is running perfectly well; the actual fix is a pip install.
            self.unavailable_reason = "driver-not-installed"
            logger.warning(
                "The 'redis' package is not installed (%s), so Redis is not being used at all — "
                "this is NOT a connection problem and the server, if running, is fine. "
                "Install it with: pip install -r requirements.txt. Using in-memory cache fallback.",
                e,
            )
            self.is_fallback = True
            self.client = self.fallback_store
        except Exception as e:
            self.unavailable_reason = "server-unreachable"
            logger.warning(
                "The 'redis' package is installed but no server is reachable at %s (%s). "
                "Using in-memory cache fallback.",
                settings.REDIS_URI, e,
            )
            self.is_fallback = True
            self.client = self.fallback_store

    @property
    def backend_name(self) -> str:
        return "in-memory-fallback" if self.is_fallback else "redis"

    async def get_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        try:
            val = await self.client.get(f"active_state:{conversation_id}")
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    async def set_state(self, conversation_id: str, state: Dict[str, Any], ttl: int = 3600):
        try:
            val_str = json.dumps(state)
            await self.client.set(f"active_state:{conversation_id}", val_str, ex=ttl)
        except Exception:
            pass

db_client = MongoDBClient()
redis_client = RedisClient()
