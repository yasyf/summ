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
        {
            "fact": "The hardest process is accounts payable, it costs us 200 man-hours.",
            "context": "We do lots of proccesses, and try to stack rack them by the amount of man hours. I've done 11 bots while I've been here. Accounts payable has been hard, but others too. I've done a lot of bespoke bots for the FDA too. They break all the time, but they save us a lot of money.",
            "attributes": [
                Classes.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR,
                Classes.DEPARTMENT_ENGINEERING,
                Classes.COMPANY_CATEGORY_CUSTOMER,
                Classes.DEPARTMENT_FINANCE,
                Classes.INDUSTRY_CONSTRUCTION,
            ],
        },
        {
            "fact": "Ugh, all the invoice stuff really sucks, and it's been expensive for the org",
            "context": "OCR is the future since invoice processing is so hard. It's hard because even Google's OCR isn't good at capturing all the handwritten letters, maybe 70 percent hit rate. Invoice has been really expensive for us, but we spend 3m+ annually on people doing invoice processing.",
            "attributes": [
                Classes.JOB_TITLE_MANAGER,
                Classes.DEPARTMENT_FINANCE,
                Classes.COMPANY_CATEGORY_CUSTOMER,
                Classes.INDUSTRY_ENERGY_UTILITIES_WASTE,
            ],
        },
        {
            "fact": "We do a lot of helping with employee onboarding. It's a classic format.",
            "context": "Many times I've been involved in like onboarding or off boarding probably like four or five.  All the appropriate permissions need to be figured out, but I also have to navigate all of the individual systems that it has to go to make those actions outside of like your central directory. You're not going to run your desk assignment out of AD So you have to get access to that system and then, you know, depending on their landscape like you have to go out to each individual like oh, you know my exchange admin to turn off this guy's mailbox.",
            "attributes": [
                Classes.JOB_TITLE_EXECUTIVE,
                Classes.COMPANY_CATEGORY_CONSULTANCY,
                Classes.DEPARTMENT_CONSULTING,
                Classes.INDUSTRY_CONSULTING,
            ],
        },
        {
            "fact": "A lot of our hard processes deal with apps that change regularly.",
            "context": "Yeah, I mean all you really need is for the web developers to have to update one selector, And you're screwed. I mean it just feels like a macro so if it can't identify, you know you can have. You're gonna have error handling and such but I mean if you're screwed.",
            "attributes": [
                Classes.JOB_TITLE_EXECUTIVE,
                Classes.COMPANY_CATEGORY_CUSTOMER,
                Classes.DEPARTMENT_IT,
                Classes.INDUSTRY_MANUFACTURING,
            ],
        },
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
