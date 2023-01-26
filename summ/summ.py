from pathlib import Path
from typing import Optional

from summ.classify.classes import Classes
from summ.pipeline import Pipeline
from summ.query.querier import Querier


class Summ:
    """The main entry point for both populating and querying the model."""

    def __init__(self, index: str = "sum-facts"):
        self.index = index

    def populate(
        self,
        path: Path,
        parallel: bool = True,
        pipe: Optional[Pipeline] = None,
    ):
        pipe = pipe or Pipeline.default(path, self.index)
        try:
            print("Creating index, this may take a while...")
            pipe.embedder.create_index()
        except Exception:
            print("Index already exists!")

        pipe.run(parallel=parallel)

    def query(
        self,
        query: str,
        n: int = 3,
        classes: list[Classes] = [],
        debug: bool = True,
    ) -> str:
        querier = Querier(index=self.index, debug=debug)
        return querier.query(query, n=n, classes=classes)
