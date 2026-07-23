import json
import types
from abc import abstractmethod
from typing import ClassVar, Optional, Self, Union

import metrohash
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from redis import Redis


class CacheDocument(BaseModel):
    """A serializable version of a Document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_doc(cls, doc: Document):
        for k, v in doc.metadata.items():
            if isinstance(v, types.GeneratorType):
                doc.metadata[k] = list(v)
        return cls(**doc.dict())


class CacheItem(BaseModel):
    """A base class for cached responses."""

    pk: Optional[str] = None
    _redis: ClassVar[Redis] = Redis()

    @classmethod
    def passthrough(cls, **kwargs) -> Self:
        instance = cls.construct(**kwargs)
        instance.pk = cls.make_pk(instance)
        if cached := cls.safe_get(instance.pk):
            return cached

        for k in cls.__fields__.keys() - kwargs.keys():
            setattr(instance, k, None)

        return instance

    @classmethod
    def safe_get(cls, pk: Optional[str]) -> Optional[Self]:
        if pk is None:
            return None

        payload = cls._redis.get(cls._cache_key(pk))
        if payload is None:
            return None
        return cls.parse_raw(payload)

    @classmethod
    def _cache_key(cls, pk: str) -> str:
        return f"summ:{cls.__module__}.{cls.__qualname__}:{pk}"

    @staticmethod
    def _hash(s: str):
        return metrohash.hash64(s, seed=0).hex()

    @classmethod
    @abstractmethod
    def make_pk(cls, instance: Self) -> str:
        raise NotImplementedError

    def save(self) -> Self:
        self.pk = self.make_pk(self)
        self._redis.set(self._cache_key(self.pk), self.json())
        return self


class ChainCacheItem(CacheItem):
    """A base class for cached langchain LLM responses."""

    klass: str
    name: str
    document: Union[CacheDocument, list[CacheDocument]]
    result: str
    meta: dict = Field(default_factory=dict)

    def page_contents(self) -> list[str]:
        if isinstance(self.document, list):
            return [doc.page_content for doc in self.document]
        return [self.document.page_content]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(
            ":".join(
                [
                    instance.klass,
                    instance.name,
                    *instance.page_contents(),
                    json.dumps(instance.meta, sort_keys=True),
                ]
            )
        )
