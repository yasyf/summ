from enum import StrEnum, auto

from summ.classify import Classes

# https://summ.readthedocs.io/en/stable/#tags


class MyClasses(Classes, StrEnum):
    # SOURCE
    SOURCE_PODCAST = auto()
    SOURCE_INTERVIEW = auto()
    SOURCE_RADIO = auto()
