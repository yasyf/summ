import itertools
from typing import Self

import metrohash
import pinecone
from langchain.docstore.document import Document as Document
from langchain.embeddings import OpenAIEmbeddings
from redis_om import EmbeddedJsonModel

from user_interview_summary.cache.cacher import CacheItem


class EmbedDocument(EmbeddedJsonModel, Document):
    pass


class Embedding(CacheItem):
    document: EmbedDocument
    fact: str
    embedding: list[float]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return metrohash.hash64(instance.fact, seed=0).hex()


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
            Embedding.passthrough(
                document=EmbedDocument(**doc.dict()),
                facts=f,
                embeddings=e,
            )
            for f, e in zip(doc.metadata["facts"], embeddings)
        ]

    def persist(self, doc: Document) -> list[Embedding]:
        embeddings = self.embed(doc)
        vectors = [
            (
                e.pk,
                e.embedding,
                {
                    "classes": list(
                        itertools.chain.from_iterable(
                            e.document.metadata["classes"].values()
                        )
                    ),
                },
            )
            for e in embeddings
        ]
        if vectors:
            self.index.upsert(vectors)
        return embeddings
