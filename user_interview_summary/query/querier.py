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
            "fact": "The hardest process is accounts payable, it costs us 200 man-hours.",
            "context": "We do lots of proccesses, and try to stack rack them by the amount of man hours. I've done 11 bots while I've been here. Accounts payable has been hard, but others too. I've done a lot of bespoke bots for the FDA too. It's hard because these bots break all the time, but they save us a lot of money.",
            "attributes": "JOB_TITLE_INDIVIDUAL_CONTRIBUTOR, DEPARTMENT_ENGINEERING, COMPANY_CATEGORY_CUSTOMER, DEPARTMENT_FINANCE, INDUSTRY_CONSTRUCTION",
        },
        {
            "fact": "Ugh, all the invoice stuff really sucks, and it's been expensive for the org",
            "context": "OCR is the future since invoice processing is so hard. It's hard because even Google's OCR isn't good at capturing all the handwritten letters, maybe 70% hit rate. Invoice has been really expensive for us, but we spend 3m+ annually on our automation doing invoice processing.",
            "attributes": "JOB_TITLE_MANAGER, DEPARTMENT_FINANCE, COMPANY_CATEGORY_CUSTOMER, INDUSTRY_ENERGY_UTILITIES_WASTE",
        },
    ]
    EXAMPLE_PROMPT = PromptTemplate(
        input_variables=["fact", "context", "attributes"],
        template=dedent(
            """
                FACT: {fact}
                CONTEXT: {context}
                ATTTRIBUTES: {attributes}
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
                You are an agent summarizing insights from a set of user interviews. You are given:
	            (1) a query that you are trying to answer
	            (2) a set of facts as well as their context and attributes about the user

                Based on the given query, your job is to read through the facts and related context and give an answer to the query.

                Here is an example:
                EXAMPLE:
                ==================================================
                {self.examples()}
                ==================================================

                The current query, and facts with their user attributes and context will follow. Reply with your response to the query.

                """
            ),
            suffix=dedent(
                """
                QUERY: {query}
                RESPONSE:"""
            ),
            input_variables=["query"],
            example_separator="\n",
        )

    def query(self, query: str, n=10, classes: list[Classes] = []):
        embedding = self.embeddings.embed_query(query)
        filter = {"$or": [{"classes": c.value} for c in classes]} if classes else None
        results = self.index.query(
            embedding, top_k=n, include_metadata=True  # type: ignore
        )["matches"]

        examples = [
            {
                "fact": e.fact,
                "context": e.document.metadata["summary"],
                "attributes": e.document.metadata["classes"].values(),
            }
            for r in results
            for e in [Embedding.safe_get(r["id"])]
            if e
        ]

        # Print the prompt
        # print(self.prompt_template(examples).format(query="What are the hardest processes?"))

        chain = LLMChain(llm=self.llm, prompt=self.prompt_template(examples))
        return chain.run(query=query)
