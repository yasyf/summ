from pathlib import Path
from typing import Optional

from langchain.docstore.document import Document

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
        """Populate the model with data from a given path.

        Args:
            path (Path): The path to the data (format depends on [Importer][summ.importers.Importer]).
            parallel (bool, optional): Whether to run the pipeline in parallel.
            pipe (Optional[Pipeline], optional): The pipeline to use. If one is not supplied, a default one will be constructed.
        """
        pipe = pipe or Pipeline.default(path, self.index)

        if not pipe.embedder.has_index():
            try:
                print("Creating index, this may take a while...")
                pipe.embedder.create_index()
            except Exception as e:
                if "already exists" not in str(e):
                    raise e
                print("Index already exists!")

        pipe.run(parallel=parallel)

    def query(
        self,
        question: str,
        n: int = 3,
        classes: list[Classes] = [],
        corpus: list[Document] = [],
        debug: bool = True,
    ) -> str:
        """
        Query a pre-populated model with a given question.

        Args:
            question (str): The question to ask.
            n (int, optional): The number of facts to use per sub-query.
            classes (list[Classes], optional): The set of tags to use as filters (AND).
            debug (bool, optional): Whether to print intermediate steps.
        """
        querier = Querier(index=self.index, debug=debug)
        return querier.query(question, n=n, classes=classes, corpus=corpus)
