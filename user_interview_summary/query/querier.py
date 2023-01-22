import itertools
from textwrap import dedent

import metrohash
import pinecone
from langchain import FewShotPromptTemplate, LLMChain, PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from user_interview_summary.classify.classes import Classes
from user_interview_summary.shared.chain import Chain


class Querier(Chain):
    INDEX = "rpa-user-interviews"
    EXAMPLES = [
    ]
    PROMPT_TEMPLATE = FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["fact", "context", "attributes"],
            template=dedent(
                """
                Fact: {fact}
                Context: {context}
                Attributes: {attributes}
                """
            ),
        ),
        prefix=dedent(
            f"""
                The following is a list of facts, and attributes about the user who said them. Each fact comes with context and attributes that describe the user. Each set of fact, context, attributes is said by a different user, talking about their experience.

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

    def __init__(self):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        print(self.INDEX)
        self.index = pinecone.Index(self.INDEX)

    def query(self, query: str, n=10, classes: list[Classes] = []):
        embedding = self.embeddings.embed_query(query)
        filter = {"$or": [{"classes": c.value} for c in classes]} if classes else None
        results = self.index.query(
            embedding,
            top_k=n,
            include_metadata=True,
            filter=filter,  # type: ignore
        )["matches"]

        chain = LLMChain(llm=self.llm, prompt=self.PROMPT_TEMPLATE)
        return chain.run(query=query)
