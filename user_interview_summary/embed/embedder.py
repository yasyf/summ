import itertools

import metrohash
import pinecone
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from user_interview_summary.cache.cacher import CacheItem


class Embedding(CacheItem):
    document: Document
    fact: str
    embedding: list[float]


class Embedder:
    INDEX = "rpa-user-interviews"
    DIMS = 1536

    @classmethod
    def create_index(cls):
        pinecone.create_index(
            cls.INDEX,
            dimension=cls.DIMS,
            metadata_config={"indexed": ["classes"]},
        )

    def __init__(self):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        self.index = pinecone.Index(self.INDEX)

    def embed(self, doc: Document) -> list[Embedding]:
        if not doc.metadata.get("facts"):
            return []
        embeddings = self.embeddings.embed_documents(doc.metadata["facts"])
        return [
            Embedding.passthrough(document=doc, facts=f, embeddings=e)
            for f, e in zip(doc.metadata["facts"], embeddings)
        ]

    def persist(self, doc: Document) -> list[Embedding]:
        embeddings = self.embed(doc)
        vectors = [
            (
                metrohash.hash64(e.fact, seed=0).hex(),
                e.embedding,
                {
                    "classes": list(
                        itertools.chain.from_iterable(
                            e.document.metadata["classes"].values()
                        )
                    ),
                    "pk": e.pk,
                },
            )
            for e in embeddings
        ]
        if vectors:
            self.index.upsert(vectors)
        return embeddings
