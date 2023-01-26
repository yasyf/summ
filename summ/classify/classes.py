from enum import StrEnum, auto, unique
from typing import Optional, Self


@unique
class Classes(StrEnum):
    """The parent class for all custom sets of tags.

    To define a set of custom tags, inherit from this class
    and use Pyhon Enum syntax to define them, using the category
    name as the prefix for each tag.

    Example:
        ```python
        class MyClasses(Classes):
            ROLE_IC = auto()
            ROLE_MANAGER = auto()

            SECTOR_TECH = auto()
            SECTOR_FINANCE = auto()
        ```
    """

    @classmethod
    def get(cls, val: str) -> Optional[Self]:
        """:meta private:"""
        val = val.strip()
        try:
            return cls(val)
        except ValueError:
            try:
                return cls[val]
            except KeyError:
                pass
