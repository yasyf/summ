from dataclasses import dataclass
from typing import Optional

from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document

from summ.classify.classes import Classes
from summ.shared.chain import Chain
from summ.shared.utils import dedent


@dataclass
class Source:
    """A chunk from a source data file."""

    file: str
    classes: list[Classes]
    chunk: str


@dataclass
class Fact:
    """An individual fact from some interview."""

    fact: str
    source: str


class Factifier(Chain):
    """Factifiers are responsible for taking a Document with a chunk,
    and extracting a list of facts."""

    DEFAULT_CONTEXT = "This is the start of the conversation."

    EXAMPLES = [
        {
            "context": "The conversation so far has covered the backround of the speaker. He is in sales at UiPath.",
            "chunk": "We had a client where they would, they had like a huge database legacy database of like their inventory in the store. Whenever they would whenever they would do any type of like inventory accounts, they would shut down for like eight hours but they wouldn't go in there and see the differences between like the database and it will take them 16 hours to do. Yes, insane. We built a bot that will go in there and do like we like to call it, auditing and reconciliation of all the inventories, as long as they gave us like a spreadsheet, and you could do it in an hour.",
            "facts": [
                "A client had a large legacy database for inventory in their store.",
                "The inventory reconciliation process would shut down the store for 8 hours.",
                "The process of reconciling the database would take 16 hours to complete.",
                "A bot was built to perform inventory auditing and reconciliation.",
                "The bot can complete the process in an hour as long as a spreadsheet is provided.",
            ],
            "new_context": " An RPA developer talks about a bot he made. The bot was created to reconcile a client's inventory database which used to take 16 hours to complete and shut down the store for 8 hours, and can now be done in an hour.",
        }
    ]

    EXAMPLE_TEMPLATE = PromptTemplate(
        template=dedent(
            """
            ---
            Context:
            {{ context }}

            Paragraph:
            {{ chunk }}

            Facts:
            - {{ facts | join("\n- ") }}

            Context:
            {{ new_context }}
            ---
            """
        ),
        input_variables=["context", "chunk", "facts", "new_context"],
        template_format="jinja2",
    )

    PROMPT_TEMPLATE = FewShotPromptTemplate(
        example_prompt=EXAMPLE_TEMPLATE,
        examples=EXAMPLES,
        input_variables=["context", "chunk"],
        prefix=dedent(
            """
            Your task is to take the context of a conversation, and a paragraph, and extract any pertinent facts from it.
            The facts should only cover new information introduced in the paragraph. The context is only for background; do not use it to generate facts.

            You will also generate a new context, by taking the old context and modifying it if needed to account for the additional paragraph. You do not need to change the old context if it is suitable; simply return it again.

            Here is an example:
            """
        ),
        suffix=dedent(
            """
            Now the real one:

            ---
            Context:
            {context}

            Paragraph:
            {chunk}

            Facts:
            -
            """
        ),
    )

    def __init__(self, *args, context: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context or self.DEFAULT_CONTEXT

    def parse(self, results: str) -> tuple[list[str], str]:
        try:
            idx = results.lower().index("context")
            facts_raw, context_raw = results[:idx], results[idx:]
            context = "\n".join(context_raw.splitlines()[1:])
        except ValueError:
            facts_raw, context = results, self.context

        facts = self._parse(facts_raw.splitlines(), prefix=r"-+")
        return facts, context

    def factify(self, doc: Document) -> list[str]:
        """Returns a list of facts from the given document."""

        chain = LLMChain(llm=self.llm, prompt=self.PROMPT_TEMPLATE)
        results = "- " + self.cached(
            "factify",
            chain,
            doc,
            lambda d: {"chunk": d.page_content, "context": self.context},
        )
        facts, self.context = self.parse(results)
        return facts
