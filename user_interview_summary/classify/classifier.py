from abc import ABC, abstractmethod
from typing import Type

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

    classifiers: dict[str, "Type[BaseClassifier]"] = {}

    def __init_subclass__(cls: "Type[BaseClassifier]", **kwargs):
        super().__init_subclass__(**kwargs)
        cls.classifiers[cls.CATEGORY] = cls

    @classmethod
    def classify_all(cls, doc: Document):
        return {c: klass().classify(doc) for c, klass in cls.classifiers.items()}

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

    def _classify(self, doc: Document) -> list[Classes]:
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template())
        results = self.cached(
            "_classify",
            chain,
            doc,
            lambda doc: {"title": doc.metadata["file"]},
        )
        return self._parse(results)

    @abstractmethod
    def classify(self, doc: Document) -> list[Classes]:
        raise NotImplementedError


class TitleClassifier(BaseClassifier):
    CATEGORY = "JOB_TITLE"
    VARS = {
        "title": "User Interview Title",
        "job_title": "Title Classification",
    }
    EXAMPLES = [
        {
            "title": "Brendan - UiPath - Sales",
            "job_title": Classes.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR,
        },
        {
            "title": "Amy - American Express - HR Manager",
            "job_title": Classes.JOB_TITLE_MANAGER,
        },
        {
            "title": "Mia - Captial One - Head of IT",
            "job_title": Classes.JOB_TITLE_EXECUTIVE,
        },
    ]
    PREFIX = "The user interviewed falls into one of the following categories:"
    SUFFIX = " ".join(
        [
            f"{Classes.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR} represents an employee who contributes to a team or organization but do not manage others.",
            f"{Classes.JOB_TITLE_MANAGER} represents an employee who takes a leadership role in an organization and manages a team of employees.",
            f"{Classes.JOB_TITLE_EXECUTIVE} represents an an employee who has a leadership role in the organization. C-suite members, as well as heads, directors, and VPs of departments count as executives.",
            "Return only a list of industry variables, no explanation. The form of the title is 'Name - Name of Company - Role'. You will have to infer the role from the company name.",
        ]
    )

    def classify(self, doc: Document) -> list[Classes]:
        return self._classify(doc)


class CompanyCategoryClassifier(BaseClassifier):
    CATEGORY = "COMPANY_CATEGORY"
    VARS = {
        "title": "User Interview Title",
        "company_category": "Company Category Classification",
    }
    EXAMPLES = [
        {
            "title": "Brendan - UiPath - Sales",
            "company_category": Classes.COMPANY_CATEGORY_RPA,
        },
        {
            "title": "Amy - Accenture - Lead Automation Developer.",
            "company_category": Classes.COMPANY_CATEGORY_CONSULTANCY,
        },
        {
            "title": "Mia - Captial One - HR Manager",
            "company_category": Classes.COMPANY_CATEGORY_CUSTOMER,
        },
    ]
    PREFIX = "The user interviewed falls into one of the following categories:"
    SUFFIX = " ".join(
        [
            f"{Classes.COMPANY_CATEGORY_RPA} represents companies that build RPA software. This includes companies like UiPath and Automation Anywhere.",
            f"{Classes.COMPANY_CATEGORY_CUSTOMER} represents a company that would be a buyer of RPA software. They have some course of business that has nothing to do with selling RPA software or services.",
            f"{Classes.COMPANY_CATEGORY_CONSULTANCY} represents a company that sells RPA services. There are developers and consultants that help COMPANY_CUSTOMER type companies get RPA up and running.",
            "Return only a list of department variables, no explanation.",
            "The form of the title is 'Name - Name of Company - Role'.",
            "You will have to infer the industry from the company name. Note if a user works at a company has an internal RPA team, they do not neccessarily work at a company that sells RPA software.",
            f"They may work for a {Classes.COMPANY_CATEGORY_CUSTOMER} company who has aleady purchased some RPA software in the past.",
        ]
    )

    def classify(self, doc: Document) -> list[Classes]:
        return self._classify(doc)


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
        return self._classify(doc)


class IndustryClassifier(BaseClassifier):
    CATEGORY = "INDUSTRY"
    VARS = {
        "title": "User Interview Title",
        "industry": "Industry Classification",
    }
    EXAMPLES = [
        {
            "title": "Brendan - UiPath - Sales",
            "industry": Classes.INDUSTRY_RPA_SOFTWARE,
        },
        {
            "title": "Amy - Accenture - Lead Automation Developer.",
            "industry": Classes.INDUSTRY_CONSULTING,
        },
    ]
    PREFIX = "The user interviewed works in one or more of the following industries:"
    SUFFIX = " ".join(
        [
            "Return only a list of department variables, no explanation.",
            f"For example: '{Classes.INDUSTRY_SOFTWARE}, {Classes.INDUSTRY_HOSPITALITY}' if the user works at a company in the software and hospitality industry.",
            "The form of the title is 'Name - Name of Company - Role'.",
            "You will have to infer the industry from the company name.",
            "Note: if a user works at a company has an internal RPA team, they do not neccessarily work at a company in the RPA Software industry, they may be in the RPA department in a non-RPA company.",
        ]
    )

    def classify(self, doc: Document) -> list[Classes]:
        return self._classify(doc)
