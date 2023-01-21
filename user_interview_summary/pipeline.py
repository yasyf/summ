from itertools import chain
from pathlib import Path
from typing import Generator, Iterable, TextIO

from langchain.docstore.document import Document

from user_interview_summary.classify.classifier import BaseClassifier
from user_interview_summary.embed.embedder import Embedder
from user_interview_summary.factify.factifier import Factifier
from user_interview_summary.shared.chain import Chain
from user_interview_summary.splitter.splitter import Splitter


class Pipeline(Chain):
    def __init__(self, persist: bool = False):
        super().__init__()
        self.splitter = Splitter()
        self.factifier = Factifier()
        self.classifier = BaseClassifier
        self.embedder = Embedder()
        self.persist = persist

    def _process_doc(self, doc: Document) -> Document:
        doc.metadata["facts"] = self.factifier.factify(doc)
        doc.metadata["classes"] = self.classifier.classify_all(doc)
        doc.metadata["embeddings"] = (
            self.embedder.persist(doc) if self.persist else self.embedder.embed(doc)
        )
        return doc

    def _process_blob(self, blob: TextIO) -> Iterable[Document]:
        docs = Splitter().split(Path(blob.name).stem, blob.read())
        return map(self._process_doc, docs)

    def _process_blobs(self, blobs: list[TextIO]) -> Iterable[Document]:
        return chain.from_iterable(map(self._process_blob, blobs))

    def run(self, blobs: list[TextIO]) -> Generator[Document, None, None]:
        yield from self._process_blobs(blobs)
