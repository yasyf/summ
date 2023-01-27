import itertools
from functools import cached_property
from typing import Generator, Self

import pinecone
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from openai.error import RateLimitError
from retry import retry

from summ.cache.cacher import CacheDocument, CacheItem
from summ.shared.utils import dedent


class Embedding(CacheItem):
    """A serializable embedding vector, representing a query.

    Always has an associated fact."""

    document: CacheDocument
    query: str
    fact: str
    embedding: list[float]

    @classmethod
    def make_pk(cls, instance: Self) -> str:
        return cls._hash(instance.query)


class Embedder:
    """Embedders are responsible for taking fully-populated Documents and embedding them,
    optionally persiting them to a vector store in the process.

    Currently, only Pinecone is supported.
    """

    GPT3_DIMS = 1536

    QUERIES = 1
    """The number of extra queries to generate per fact."""

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
        """Creates the named index in Pinecone."""

        pinecone.create_index(
            self.index_name,
            dimension=self.dims,
            metadata_config={"indexed": ["classes"]},
        )

    def has_index(self):
        """Checks if the named index in Pinecone exists."""

        try:
            pinecone.describe_index(self.index_name)
            return True
        except pinecone.exceptions.NotFoundException:
            return False

    def __init__(self, index: str, dims: int = GPT3_DIMS):
        """Creates a new Embedder.

        Args:
            index: The name of the vector db index to use.
            dims: The number of dimensions of the vector db index.
        """
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

    @retry(exceptions=RateLimitError, tries=5, delay=6, jitter=(0, 4))
    def _generate_query(self, fact: str, doc: Document) -> str:
        return self.query_chain.run(fact=fact, context=doc.metadata["summary"])

    def embed(
        self, doc: Document, gen_queries: bool = False
    ) -> Generator[Embedding, None, None]:
        """Yields a set of embeddings for a given document."""

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
        """Collects the set of embeddings for a Document,
        and persists them to the vector store."""

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
