from typing import Sequence, cast

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class UnsharedDictList(Sequence):
    def __init__(self, d: dict, n: int) -> None:
        super().__init__()
        self._d = d
        self._n = n

    def __getitem__(self, _n: int):
        return self._d.copy()

    def __len__(self):
        return self._n


class Splitter:
    """Splitters are responsible for taking a file and splitting it into a list of documents (chunks).

    By defauly, we just split on double-newlines (paragraphs).
    """

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,
            chunk_overlap=0,
        )

    def get_chunks(self, title: str, text: str) -> list[str]:
        return text.split("\n\n")

    def split(self, title: str, text: str):
        chunks = self.get_chunks(title, text)
        return self.splitter.create_documents(
            chunks, cast(list, UnsharedDictList({"file": title}, len(chunks)))
        )
