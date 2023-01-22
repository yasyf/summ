import itertools
from typing import Generator, Self

import metrohash
import pinecone
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from user_interview_summary.cache.cacher import CacheDocument, CacheItem


class Embedding(CacheItem):
    document: CacheDocument
    fact: str
    embedding: list[float]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(instance.fact)


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

    def embed(self, doc: Document) -> Generator[Embedding, None, None]:
        for fact in doc.metadata["facts"]:
            embedding = Embedding.passthrough(fact=fact)
            if not embedding.embedding:
                embedding.document = CacheDocument(**doc.dict())
                embedding.embeddings = self.embeddings.embed_documents([fact])[0]
                embedding.save()
            yield embedding

    def persist(self, doc: Document) -> list[Embedding]:
        embeddings = list(self.embed(doc))
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
