from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from summ.splitter.splitter import Splitter


class GPTSplitter(Splitter):
    """Split to ensure a GPT prompt is not too long."""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2500,
            chunk_overlap=200,
        )

    def get_chunks(self, title: str, text: str) -> list[str]:
        return [text]
