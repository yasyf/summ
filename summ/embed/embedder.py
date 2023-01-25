import itertools
from functools import cached_property
from typing import Generator, Self

import metrohash
import pinecone
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from summ.cache.cacher import CacheDocument, CacheItem
from summ.shared.utils import dedent


class Embedding(CacheItem):
    document: CacheDocument
    query: str
    fact: str
    embedding: list[float]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(instance.query)


class Embedder:
    GPT3_DIMS = 1536
    QUERIES = 1
    QUERY_TEMPLATE = PromptTemplate(
        input_variables=["fact", "context"],
        template=dedent(
            """
            A user was interviewed, and stated a fact. Given this fact and the context of the interview, create a question that this fact is the answer to. The question should be specific to this fact.

            Fact: {fact}
            Context: {context}
            Question:
            """
        ),
    )

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

    def _embed(self, query: str, fact: str, doc: Document) -> Embedding:
        embedding = Embedding.passthrough(query=query)

        if not embedding.embedding:
            embedding.fact = fact
            embedding.document = CacheDocument.from_doc(doc)
            embedding.embedding = self.embeddings.embed_documents([query])[0]
            embedding.save()

        return embedding

    @cached_property
    def query_chain(self):
        return LLMChain(
            llm=OpenAI(temperature=0.7, cache=False), prompt=self.QUERY_TEMPLATE
        )

    def _generate_query(self, fact: str, doc: Document) -> str:
        return self.query_chain.run(fact=fact, context=doc.metadata["summary"])

    def embed(
        self, doc: Document, gen_queries: bool = False
    ) -> Generator[Embedding, None, None]:
        for fact in doc.metadata["facts"]:
            yield self._embed(query=fact, fact=fact, doc=doc)
            if gen_queries:
                for _ in range(self.QUERIES):
                    yield self._embed(
                        query=self._generate_query(fact, doc),
                        fact=fact,
                        doc=doc,
                    )

    def persist(self, doc: Document) -> list[Embedding]:
        embeddings = list(self.embed(doc, gen_queries=True))
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
