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
            "chunk": "Well, it's a mix in the case of like, for example, invoices, so, sorry, in the case of currency there's an ML model, just for that, but but it's a simpler one. It's not like a deep learning one because it's not necessary for that and the compute cost is like customers don't have. They don't want to do and don't want to spend money on GPUs left and right, because they, they don't afford",
            "facts": [
                "At UiPath, there is an ML model for currency.",
                "The ML model for currency at UiPath is not a deep learning one.",
                "Customers of UiPath do not want to spend money on GPUs.",
            ],
            "new_context": "The conversation so far has detailed the ML model for currency at UiPath, as told by a salesperson.",
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
