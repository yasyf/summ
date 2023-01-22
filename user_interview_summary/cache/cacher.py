from abc import abstractmethod
from typing import Self, cast

import metrohash
from langchain.docstore.document import Document
from redis_om import EmbeddedJsonModel, JsonModel, NotFoundError


class CacheDocument(EmbeddedJsonModel, Document):
    pass


class CacheItem(JsonModel):
    @classmethod
    def passthrough(cls, *args, **kwargs) -> Self:
        instance = cls(*args, **kwargs)
        instance.pk = cls.make_pk(instance)
        try:
            return cast(Self, cls.get(instance.pk))
        except NotFoundError:
            return cast(Self, instance.save())

    @staticmethod
    def _hash(s: str):
        return metrohash.hash64(s, seed=0).hex()

    @classmethod
    @abstractmethod
    def make_pk(cls, instance: Self) -> str:
        raise NotImplementedError


class ChainCacheItem(CacheItem):
    klass: str
    name: str
    document: CacheDocument
    result: str

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(
            ":".join([instance.klass, instance.name, instance.document.page_content])
        )
