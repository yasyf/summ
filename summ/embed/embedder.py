import itertools
from typing import Generator, Self

import metrohash
import pinecone
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from summ.cache.cacher import CacheDocument, CacheItem


class Embedding(CacheItem):
    document: CacheDocument
    fact: str
    embedding: list[float]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(instance.fact)


class Embedder:
    GPT3_DIMS = 1536

    def create_index(self):
        pinecone.create_index(
            self.index_name,
            dimension=self.dims,
            metadata_config={"indexed": ["classes"]},
        )

    def __init__(self, index: str, dims: int = GPT3_DIMS):
        super().__init__()
        self.index_name = index
        self.dims = dims
        self.embeddings = OpenAIEmbeddings()
        self.index = pinecone.Index(index)

    def embed(self, doc: Document) -> Generator[Embedding, None, None]:
        for fact in doc.metadata["facts"]:
            embedding = Embedding.passthrough(fact=fact)

            if not embedding.embedding:
                embedding.document = CacheDocument.from_doc(doc)
                embedding.embedding = self.embeddings.embed_documents([fact])[0]
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
