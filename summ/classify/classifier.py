from abc import ABC, abstractmethod
from typing import Generic, Self, Type, TypeVar, cast

from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document

from summ.classify.classes import Classes
from summ.shared.chain import Chain, TExtract
from summ.shared.utils import dedent

C = TypeVar("C", bound=Classes)


class Classifier(ABC, Generic[C], Chain):
    CATEGORY: str
    EXAMPLES: list[dict[str, str]]
    VARS: dict[str, str]
    PREFIX: str
    SUFFIX: str

    classes: Type[C]
    classifiers: dict[str, Type[Self]] = {}

    def __init_subclass__(cls: Type[Self], classes: Type[C], **kwargs):
        super().__init_subclass__(**kwargs)
        cls.classes = classes
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
            [c.value for c in self.classes if c.startswith(self.CATEGORY.lower())]
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

    def _parse(self, results: str) -> list[C]:
        return [
            c for result in results.split(",") for c in [self.classes.get(result)] if c
        ]

    def classify(self, doc: Document) -> list[C]:
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template())
        results = self.cached(
            "_classify",
            chain,
            doc,
            self._classify,
        )
        return self._parse(results)

    @abstractmethod
    def _classify(self, doc: Document) -> dict[str, str]:
        raise NotImplementedError
