from pathlib import Path
from typing import Iterable, TextIO

from langchain.docstore.document import Document

from summ.splitter.gpt_splitter import GPTSplitter


class Importer:
    """Importers are responsible for extracting file-like buffers from a data source."""

    def __init__(self, dir: Path):
        """A default importer which reads from a directory of text files."""

        self.dir = dir

    @property
    def paths(self) -> Iterable[Path]:
        return self.dir.glob("*.txt")

    @property
    def blobs(self) -> Iterable[TextIO]:
        return map(Path.open, self.paths)

    def docs(self) -> Iterable[Document]:
        return [Document(page_content=blob.read()) for blob in self.blobs]
