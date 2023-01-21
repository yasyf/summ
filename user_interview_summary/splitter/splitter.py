from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
        return self.splitter.create_documents(chunks, [{"file": title}] * len(chunks))
