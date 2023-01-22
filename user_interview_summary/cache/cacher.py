from typing import Self, cast

from redis_om import HashModel, NotFoundError


class CacheItem(HashModel):
    @classmethod
    def passthrough(cls, *args, **kwargs) -> Self:
        instance = cls(*args, **kwargs)
        try:
            return cast(Self, cls.get(instance.pk))
        except NotFoundError:
            return cast(Self, instance.save())
