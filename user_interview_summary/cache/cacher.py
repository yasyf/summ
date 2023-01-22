from abc import abstractmethod
from typing import Self, cast

from redis_om import JsonModel, NotFoundError


class CacheItem(JsonModel):
    @classmethod
    def passthrough(cls, *args, **kwargs) -> Self:
        instance = cls(*args, **kwargs)
        instance.pk = cls.make_pk(instance)
        try:
            return cast(Self, cls.get(instance.pk))
        except NotFoundError:
            return cast(Self, instance.save())

    @classmethod
    @abstractmethod
    def make_pk(cls, instance: Self) -> str:
        raise NotImplementedError
