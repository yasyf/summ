from pathlib import Path
from typing import Optional

from langchain.docstore.document import Document

from summ.classify.classes import Classes
from summ.embed.embedder import Embedder
from summ.pipeline import Pipeline
from summ.query.querier import Querier


class Summ:
    """The main entry point for both populating and querying the model."""

    def __init__(self, index: str = "sum-facts", n: int = 3):
        self.index = index
        self.n = n

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
                pipe.dprint("Create Index", pipe.embedder.index_name)
                pipe.embedder.create_index()
            except Exception as e:
                if "already exists" in str(e):
                    msg = "Index already exists!"
                elif "quota" in str(e):
                    msg = "You have exceeded the number of indexes for your Pinecone tier!"
                else:
                    raise e
                print(msg)
                pipe.dprint(msg)
        else:
            pipe.dprint("Index already exists!")

        pipe.run(parallel=parallel)

    def query(
        self,
        question: str,
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
        if not Embedder(self.index).has_index():
            raise Exception(
                f"Index {self.index} not found! Please run `summ populate` first."
            )
        querier = Querier(index=self.index, debug=debug)
        return querier.query(question, n=self.n, classes=classes, corpus=corpus)
