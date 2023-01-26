from dataclasses import dataclass

from langchain import PromptTemplate
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

    PROMPT_TEMPLATE = PromptTemplate(
        template=dedent(
            """
            Your task is to take a paragraph, and extract any pertinent facts from it.
            The facts should be formatted in a bulleted list.

            Paragraph:
            {chunk}

            Facts:
            -
            """
        ),
        input_variables=["chunk"],
    )

    def parse(self, results: list[str]):
        return self._parse(results, prefix=r"-+")

    def factify(self, doc: Document) -> list[str]:
        """Returns a list of facts from the given document."""

        chain = LLMChain(llm=self.llm, prompt=self.PROMPT_TEMPLATE)
        results = "-" + self.cached("factify", chain, doc)
        return self.parse(results.splitlines())
