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
    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,
            chunk_overlap=0,
        )

    def split(self, title: str, text: str):
        chunks = [
            speaker_chunk[1]
            for utterance in text.split("\n\n")
            for speaker_chunk in [utterance.split("\n")]
            if "\n" in utterance and "markie" not in speaker_chunk[0].lower()
        ]
        return self.splitter.create_documents(
            chunks, cast(list, UnsharedDictList({"file": title}, len(chunks)))
        )
