from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Splitter:
    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,
            chunk_overlap=0,
        )

    def split(self, title: str, text: str):
        return self.splitter.create_documents([text], [{"file": title}])
