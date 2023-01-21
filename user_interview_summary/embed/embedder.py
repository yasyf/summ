import os
from dataclasses import dataclass

import metrohash
import pinecone
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from user_interview_summary.shared.chain import Chain

if pinecone_api_key := os.environ.get("PINECONE_API_KEY"):
    pinecone.init(api_key=pinecone_api_key)


@dataclass
class Embedding:
    document: Document
    fact: str
    embedding: list[float]


class Embedder(Chain):
    INDEX = "rpa-user-interviews"

    def __init__(self):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        self.index = pinecone.Index(self.INDEX)
        self.hasher = pyhash.fnv1_32()

    def embed(self, doc: Document) -> list[Embedding]:
        embeddings = self.embeddings.embed_documents([doc.metadata["facts"]])
        return [Embedding(doc, f, e) for f, e in zip(doc.metadata["facts"], embeddings)]

    def persist(self, doc: Document) -> list[Embedding]:
        embeddings = self.embed(doc)
        self.index.upsert(
            [
                (
                    metrohash.hash64_int(e.fact, seed=0),
                    e.embedding,
                    {
                        "title": e.document.metadata["title"],
                        "fact": e.fact,
                        "classes": e.document.metadata["classes"],
                        "document": e.document.page_content,
                    },
                )
                for e in embeddings
            ]
        )
        return embeddings
