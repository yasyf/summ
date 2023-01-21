from abc import ABC, abstractmethod

from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

from user_interview_summary.classify.classes import Classes
from user_interview_summary.shared.chain import Chain
from user_interview_summary.shared.utils import dedent


class BaseClassifier(ABC, Chain):
    CATEGORY: str
    EXAMPLES: list[dict[str, str]]
    VARS: dict[str, str]
    PREFIX: str
    SUFFIX: str

    def example_template(self, dynamic=set()) -> str:
        return "\n".join(
            [
                f"{v}: { '' if k in dynamic else '{' + k + '}' }"
                for k, v in self.VARS.items()
            ]
        )

    def prompt_template(self):
        classes = "\n".join(
            [c.value for c in Classes if c.startswith(self.CATEGORY.lower())]
        )
        return FewShotPromptTemplate(
            examples=self.EXAMPLES,
            example_prompt=PromptTemplate(
                input_variables=list(self.VARS.keys()),
                template=self.example_template(),
            ),
            prefix=dedent(
                f"""
                The following is a title of a user interview. {self.PREFIX}
                {classes}

                {self.SUFFIX}

                """
            ),
            suffix=self.example_template(dynamic={self.CATEGORY.lower()}),
            input_variables=list(self.VARS.keys() - {self.CATEGORY.lower()}),
            example_separator="\n",
        )

    def debug_prompt(self, **kwargs: dict[str, str]) -> str:
        return self.prompt_template().format(**kwargs)

    def _parse(self, results: str) -> list[Classes]:
        return [c for result in results.split(",") for c in [Classes.get(result)] if c]

    def _classify(self, **kwargs: dict[str, str]) -> list[Classes]:
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template())
        results = chain.run(**kwargs)
        return self._parse(results)

    @abstractmethod
    def classify(self, doc: Document) -> list[Classes]:
        raise NotImplementedError


class DepartmentClassifier(BaseClassifier):
    CATEGORY = "DEPARTMENT"
    VARS = {
        "title": "User Interview Title",
        "department": "Department Classification",
    }
    EXAMPLES = [
        {"title": "Brendan - UiPath - Sales", "department": Classes.DEPARTMENT_SALES},
        {
            "title": "Amy - Blue Prism - Lead Automation Developer",
            "department": Classes.DEPARTMENT_RPA_DEVELOPMENT,
        },
    ]
    PREFIX = "The user interviewed works in one or more of the following departments:"
    SUFFIX = " ".join(
        [
            "Return only a list of department variables, no explanation.",
            f"For example: '{Classes.DEPARTMENT_OPERATIONS}, {Classes.DEPARTMENT_SALES}' if the user works in both Operations and Sales.",
            "The form of the title is 'Name - Name of Company - Role'.",
            "Note if a user works at a company that builds RPA software or is an RPA consultancy, they do not work at the RPA Center of Excellence; it means they are an RPA developer to external customers.",
        ]
    )

    def classify(self, doc: Document) -> list[Classes]:
        return self._classify(title=doc.metadata["file"])
