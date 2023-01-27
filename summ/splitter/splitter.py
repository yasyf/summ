import os
from typing import Self, Sequence, Type, cast

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

    @classmethod
    def wrap(cls: Type[Self], other: "Splitter") -> "Splitter":
        """Wrap an existing splitter to chain processing."""

        class WrappedSplitter(cls):  # type: ignore
            def split(self, title: str, text: str):
                docs = other.split(title, text)
                prefix = os.path.commonprefix([doc.page_content for doc in docs])
                return super().split(
                    title,
                    "\n\n".join(doc.page_content.removesuffix(prefix) for doc in docs),
                )

        return WrappedSplitter()

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
