import re
from dataclasses import dataclass
from textwrap import dedent

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document

from summ.classify.classes import Classes
from summ.shared.chain import Chain


@dataclass
class Source:
    file: str
    classes: list[Classes]
    chunk: str


@dataclass
class Fact:
    fact: str
    source: str


class Factifier(Chain):
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
        chain = LLMChain(llm=self.llm, prompt=self.PROMPT_TEMPLATE)
        results = "-" + self.cached("factify", chain, doc)
        return self.parse(results.splitlines())
