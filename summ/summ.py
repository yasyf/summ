from pathlib import Path

from summ.classify.classes import Classes
from summ.embed.embedder import Embedder
from summ.pipeline import Pipeline
from summ.query.querier import Querier


class Summ:
    def __init__(self, index: str = "sum-facts"):
        self.index = index

    def populate(self, path: Path, parallel: bool = True):
        pipe = Pipeline.default(path, self.index)
        try:
            pipe.embedder.create_index()
        except Exception:
            pass

        pipe.run(parallel=parallel)

    def query(
        self,
        query: str,
        n: int = 3,
        classes: list[Classes] = [],
        debug: bool = True,
    ):
        querier = Querier(index=self.index, debug=debug)
        querier.query(query, n=n, classes=classes)
