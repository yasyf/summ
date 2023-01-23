import logging
import os
import traceback
import types
from itertools import chain
from pathlib import Path
from typing import Callable, Generator, Iterable, ParamSpec, TextIO, TypeVar

from joblib import Parallel, delayed
from langchain.docstore.document import Document

from user_interview_summary.classify.classifier import BaseClassifier
from user_interview_summary.embed.embedder import Embedder
from user_interview_summary.factify.factifier import Factifier
from user_interview_summary.shared.chain import Chain
from user_interview_summary.splitter.splitter import Splitter
from user_interview_summary.summarize.summarizer import Summarizer

T = TypeVar("T")
R = TypeVar("R")


class Pipeline(Chain):
    def __init__(self, persist: bool = False):
        super().__init__()
        self.splitter = Splitter()
        self.factifier = Factifier()
        self.classifier = BaseClassifier
        self.embedder = Embedder()
        self.summarizer = Summarizer()
        self.persist = persist
        self.pool = Parallel(n_jobs=8, prefer="threads", verbose=10)

    def _process_doc(self, doc: Document) -> Document:
        try:
            doc.metadata["facts"] = self.factifier.factify(doc)
            doc.metadata["classes"] = self.classifier.classify_all(doc)
            doc.metadata["summary"] = self.summarizer.summarize_doc(doc)
            doc.metadata["embeddings"] = (
                self.embedder.persist(doc) if self.persist else self.embedder.embed(doc)
            )
        except Exception as e:
            logging.error(f"Error processing {doc.metadata['file']}")
            traceback.print_exception(e)
            if "PYTEST_CURRENT_TEST" in os.environ:
                raise e
        finally:
            return doc

    def _split_blob(self, blob: TextIO) -> Iterable[Document]:
        return self.splitter.split(Path(blob.name).stem, blob.read())

    def _process_blob(self, blob: TextIO) -> Iterable[Document]:
        return map(self._process_doc, self._split_blob(blob))

    def _parallel(self, meth: Callable[[T], R], it: Iterable[T]) -> Iterable[R]:
        return self.pool(delayed(meth)(x) for x in it) or []

    def run(self, blobs: Iterable[TextIO]) -> Generator[Document, None, None]:
        yield from chain.from_iterable(map(self._process_blob, blobs))

    def _runp(self, blobs: Iterable[TextIO]) -> Generator[Document, None, None]:
        with self.pool:
            for docs in self._parallel(self._split_blob, blobs):
                yield from self._parallel(self._process_doc, docs)

    def runp(self, blobs: Iterable[TextIO]) -> Iterable[Document]:
        return list(self._runp(blobs))
