import re
from dataclasses import dataclass
from textwrap import dedent

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

from user_interview_summary.classify.classes import Classes


@dataclass
class Source:
    file: str
    classes: list[Classes]
    chunk: str


@dataclass
class Fact:
    fact: str
    source: str


class Factifier:
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

    def __init__(self) -> None:
        self.llm = OpenAI(temperature=0.0)

    def _parse(self, results: list[str]):
        return [
            p.group("fact")
            for r in results
            for p in [re.search(r"-(?:\s*)(?P<fact>.*)", r)]
            if p
        ]

    def factify(self, doc: Document) -> list[Document]:
        chain = LLMChain(llm=self.llm, prompt=self.PROMPT_TEMPLATE)
        results = chain.run(doc.page_content)
        return [
            Document(page_content=d, metadata=doc.metadata)
            for d in self._parse(results.splitlines())
        ]
