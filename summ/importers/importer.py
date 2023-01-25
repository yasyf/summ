from abc import ABC
from pathlib import Path
from typing import Iterable, TextIO


class Importer:
    def __init__(self, dir: Path):
        self.dir = dir

    @property
    def paths(self) -> Iterable[Path]:
        return self.dir.glob("*.txt")

    @property
    def blobs(self) -> Iterable[TextIO]:
        return map(Path.open, self.paths)
