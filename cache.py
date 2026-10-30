
import json
import uuid

import numpy as np
import redis


class NumpyEncoder(json.JSONEncoder):
    """Serialize NumPy values into JSON-compatible types."""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


class RedisSemanticCache:
    def __init__(
        self,
        settings_or_client,
        embeddings=None,
        namespace="rag",
        ttl=86400,
    ):
        """
        Initialize the cache.

        Supports either:
        - RedisSemanticCache(redis_client)
        - RedisSemanticCache(settings, embeddings)
        """
        if isinstance(settings_or_client, redis.Redis):
            self.r = settings_or_client
            self.s = None
        else:
            self.s = settings_or_client

            redis_url = getattr(
                self.s,
                "redis_url",
                "redis://localhost:6379/0",
            )

            self.r = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
            )

        self.embeddings = embeddings
        self.ns = namespace
        self.ttl = getattr(self.s, "cache_ttl", ttl)

    def store(self, question, model, entry):
        """Store a response with an expiration time."""
        cache_entry = dict(entry) if isinstance(entry, dict) else {
            "answer": str(entry)
        }

        cache_entry["question"] = question
        cache_entry["model"] = model

        cache_key = f"{self.ns}:sem:{uuid.uuid4().hex}"

        self.r.setex(
            cache_key,
            self.ttl,
            json.dumps(cache_entry, cls=NumpyEncoder),
        )

    def lookup(self, question, model):
        """Return (cached_result, hit_kind) or (None, None)."""
        try:
            keys = self.r.scan_iter(
                match=f"{self.ns}:sem:*"
            )

            for key in keys:
                data = self.r.get(key)

                if not data:
                    continue

                entry = json.loads(data)

                if (
                    entry.get("question") == question
                    and entry.get("model", model) == model
                ):
                    result = dict(entry)
                    result.pop("question", None)
                    result.pop("model", None)

                    return result, "exact"

            return None, None

        except (redis.RedisError, json.JSONDecodeError, TypeError):
            return None, None

    def stats(self):
        """Return the number of cache entries."""
        try:
            return sum(
                1
                for _ in self.r.scan_iter(
                    match=f"{self.ns}:sem:*"
                )
            )
        except redis.RedisError:
            return 0
