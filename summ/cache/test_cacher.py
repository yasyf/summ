from typing import Optional

from summ.cache.cacher import CacheDocument, ChainCacheItem


class FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


def test_passthrough_round_trips_complete_cache_item(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(ChainCacheItem, "_redis", redis)

    item = ChainCacheItem.passthrough(
        klass="Summarizer",
        name="summarize",
        document=CacheDocument(page_content="text"),
        meta={"style": "short"},
    )
    assert item.result is None
    assert redis.values == {}

    item.result = "summary"
    item.save()

    cached = ChainCacheItem.passthrough(
        klass="Summarizer",
        name="summarize",
        document=CacheDocument(page_content="text"),
        meta={"style": "short"},
    )
    assert cached.result == "summary"
    assert cached.pk == item.pk


def test_cache_keys_are_scoped_to_model_type():
    pk = "same"
    assert ChainCacheItem._cache_key(pk).endswith(f"ChainCacheItem:{pk}")
