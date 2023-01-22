import itertools
from typing import cast

import metrohash
import pinecone
from langchain import FewShotPromptTemplate, LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings

from user_interview_summary.classify.classes import Classes
from user_interview_summary.embed.embedder import Embedding
from user_interview_summary.shared.chain import Chain
from user_interview_summary.shared.utils import dedent


class Querier(Chain):
    INDEX = "rpa-user-interviews"
    EXAMPLES = [
        {
            "fact": "I like to play video games",
            "attributes": ["gamer"],
            "context": "I like to play video games",
        },
    ]
    EXAMPLE_PROMPT = PromptTemplate(
        input_variables=["fact", "context", "attributes"],
        template=dedent(
            """
                Fact: {fact}
                Context: {context}
                Attributes: {attributes}
                """
        ),
    )

    def __init__(self):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        self.index = pinecone.Index(self.INDEX)

    def examples(self):
        return FewShotPromptTemplate(
            examples=self.EXAMPLES,
            example_prompt=self.EXAMPLE_PROMPT,
            prefix="",
            suffix="",
            input_variables=[],
            example_separator="\n",
        ).format()

    def prompt_template(self, examples):
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.EXAMPLE_PROMPT,
            prefix=dedent(
                f"""
                The following is a list of facts, and attributes about the user who said them. Each fact comes with context and attributes that describe the user. Each set of fact, context, attributes is said by a different user, talking about their experience.

                Here are some examples:
                {self.examples()}

                """
            ),
            suffix=dedent(
                """
            Can you write a long, multi-paragraph summary the facts here and pull in context when neccessary to answer the query: {query}
            """
            ),
            input_variables=["query"],
            example_separator="\n",
        )

    def query(self, query: str, n=10, classes: list[Classes] = []):
        embedding = self.embeddings.embed_query(query)
        filter = {"$or": [{"classes": c.value} for c in classes]} if classes else None
        results = self.index.query(
            embedding, top_k=n, filter=filter, include_metadata=True  # type: ignore
        )["matches"]

        examples = [
            {
                "fact": e.fact,
                "context": e.document.metadata["summary"],
                "attributes": e.document.metadata["classes"].values(),
            }
            for r in results
            for e in [cast(Embedding, Embedding.get(r["id"]))]
        ]

        chain = LLMChain(llm=self.llm, prompt=self.prompt_template(examples))
        return chain.run(query=query)
