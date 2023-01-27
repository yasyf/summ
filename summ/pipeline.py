import logging
import os
import traceback
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Generator, Generic, Iterable, Self, TextIO, Type

from langchain.docstore.document import Document
from openai.error import RateLimitError
from retry import retry

from summ.classify.classifier import C, Classifier
from summ.embed.embedder import Embedder
from summ.factify.factifier import Factifier
from summ.importers.importer import Importer
from summ.shared.chain import Chain
from summ.splitter.gpt_splitter import GPTSplitter
from summ.splitter.splitter import Splitter
from summ.summarize.summarizer import Summarizer


class Pipeline(Chain, Generic[C]):
    """The end-to-end population pipeline.

    This class will:
    - Take an Importer and yield a set of file-like objects.
    - Split the file-like objects into a set of chunks with a Splitter.
    - Extract facts from each chunk with a Factifier.
    - Extract a summary from each chunk with a Summarizer.
    - Embed, and optionally persist, each chunk with an Embedder.
    """

    importer: Importer
    embedder: Embedder

    @classmethod
    def default(cls, path: Path, index: str) -> Self:
        return cls(
            importer=Importer(path),
            embedder=Embedder(index),
            persist=True,
            verbose=True,
        )

    def __init__(
        self,
        importer: Importer,
        embedder: Embedder,
        persist: bool = False,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.splitter = Splitter()
        self.factifier = Factifier()
        self.classifier = Classifier[C]
        self.summarizer = Summarizer()
        self.embedder = embedder
        self.importer = importer
        self.persist = persist

    @retry(
        exceptions=RateLimitError,
        tries=5,
        delay=10,
        backoff=2,
        max_delay=120,
        jitter=(0, 10),
    )
    def _process_doc(self, doc: Document, classes: dict[str, list[C]]) -> Document:
        try:
            if "classes" not in doc.metadata:
                doc.metadata["classes"] = classes
            if "facts" not in doc.metadata:
                doc.metadata["facts"] = self.factifier.factify(doc)
            if "summary" not in doc.metadata:
                doc.metadata["summary"] = self.summarizer.summarize_doc(doc)
            if "embeddings" not in doc.metadata:
                doc.metadata["embeddings"] = (
                    self.embedder.persist(doc)
                    if self.persist
                    else self.embedder.embed(doc)
                )
        except Exception as e:
            logging.error(f"Error processing {doc.metadata['file']}")
            traceback.print_exception(e)
            if "PYTEST_CURRENT_TEST" in os.environ:
                raise e
        finally:
            return doc

    def _split_blob(self, blob: TextIO) -> list[Document]:
        return self.splitter.split(Path(blob.name).stem, blob.read())

    def _process_blob(self, blob: TextIO) -> Iterable[Document]:
        docs = self._split_blob(blob)
        classes = self.classifier.classify_all(docs)
        return map(partial(self._process_doc, classes=classes), docs)

    def _rung(self, blobs: Iterable[TextIO]) -> Generator[Document, None, None]:
        yield from chain.from_iterable(map(self._process_blob, blobs))

    def _runpg(self, blobs: Iterable[TextIO]) -> Generator[Document, None, None]:
        for docs in self._pmap(self._split_blob, blobs):
            classes = self.classifier.classify_all(docs)
            yield from self._pmap(self._process_doc, docs, classes)

    def _runp(self, blobs: Iterable[TextIO]) -> list[Document]:
        return list(self._runpg(blobs))

    def corpus(self) -> Generator[Document, None, None]:
        """Yields the extracted source corpus"""
        self.splitter = GPTSplitter.wrap(self.splitter)
        for docs in self._pmap(self._split_blob, self.importer.blobs):
            yield from docs

    def rung(self) -> Generator[Document, None, None]:
        """Yields one Embedding at a time.

        Helpful for when you want to test only a small part of your pipeline.
        """

        yield from self._rung(self.importer.blobs)

    def runp(self) -> list[Document]:
        """Calculates all embeddings in parallel. Very fast!"""

        return self._runp(self.importer.blobs)

    def run(self, parallel: bool = True) -> list[Document]:
        return self.runp() if parallel else list(self.rung())
