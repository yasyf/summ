from enum import StrEnum, auto, unique
from typing import Optional, Self


@unique
class Classes(StrEnum):
    @classmethod
    def get(cls, val: str) -> Optional[Self]:
        val = val.strip()
        try:
            return cls(val)
        except ValueError:
            try:
                return cls[val]
            except KeyError:
                pass
