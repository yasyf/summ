import itertools
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

    def embed(self, doc: Document) -> list[Embedding]:
        if not doc.metadata.get("facts"):
            return []
        embeddings = self.embeddings.embed_documents(doc.metadata["facts"])
        return [Embedding(doc, f, e) for f, e in zip(doc.metadata["facts"], embeddings)]

    def persist(self, doc: Document) -> list[Embedding]:
        embeddings = self.embed(doc)
        vectors = [
            (
                metrohash.hash64(e.fact, seed=0).hex(),
                e.embedding,
                {
                    "file": e.document.metadata["file"],
                    "fact": e.fact,
                    "classes": list(
                        itertools.chain.from_iterable(
                            e.document.metadata["classes"].values()
                        )
                    ),
                    "summary": e.document.metadata["summary"],
                    "document": e.document.page_content,
                },
            )
            for e in embeddings
        ]
        if vectors:
            self.index.upsert(vectors)
        return embeddings
