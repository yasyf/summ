import types
from abc import abstractmethod
from typing import Optional, Self, Union, cast

import metrohash
from langchain.docstore.document import Document
from pydantic import Field
from redis_om import EmbeddedJsonModel, JsonModel, NotFoundError
from typing_extensions import override


class CacheDocument(EmbeddedJsonModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_doc(cls, doc: Document):
        for k, v in doc.metadata.items():
            if isinstance(v, types.GeneratorType):
                doc.metadata[k] = list(v)
        return cls(**doc.dict())


class CacheItem(JsonModel):
    @classmethod
    def passthrough(cls, *args, **kwargs) -> Self:
        instance = cls.construct(*args, **kwargs)
        instance.pk = cls.make_pk(instance)
        if cached := cls.safe_get(instance.pk):
            return cached
        try:
            return cast(Self, cls(*args, **kwargs).save())
        except Exception:
            return instance

    @classmethod
    def safe_get(cls, pk: Optional[str]) -> Optional[Self]:
        try:
            return cast(Self, cls.get(pk))
        except NotFoundError:
            return None

    @staticmethod
    def _hash(s: str):
        return metrohash.hash64(s, seed=0).hex()

    @classmethod
    @abstractmethod
    def make_pk(cls, instance: Self) -> str:
        raise NotImplementedError

    @override
    def save(self, *args, **kwargs) -> "JsonModel":
        self.pk = self.make_pk(self)
        return super().save(*args, **kwargs)


class ChainCacheItem(CacheItem):
    klass: str
    name: str
    document: Union[CacheDocument, list[CacheDocument]]
    result: Optional[str] = None

    def page_contents(self) -> list[str]:
        if isinstance(self.document, list):
            return [doc.page_content for doc in self.document]
        return [self.document.page_content]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(
            ":".join([instance.klass, instance.name, *instance.page_contents()])
        )
