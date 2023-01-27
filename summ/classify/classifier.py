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
    """The base class for specifying custom classifiers to apply tags to an interview."""

    CATEGORY: str
    """The name of the category to tag. Must be the prefix of a set of tags in your [`Classes`][summ.classify.Classes] subclass."""

    VARS: dict[str, str]
    """A dictionary mapping variable names to descriptions. These will be used to generate the prompt."""

    EXAMPLES: list[dict[str, str]]
    """A list of few-shot examples. Each example should be a dictionary with keys corresponding to the keys in `VARS`."""

    PREFIX: str = "Your job is to classify the following interview into one of the following categories."
    """The prompt prefix."""

    SUFFIX: str = ""
    """The prompt suffix."""

    classes: Type[C]
    """A subclass of [`Classes`][summ.classify.Classes] that defines the set of tags to use."""

    classifiers: dict[str, Type[Self]] = {}
    """A registry of subclasses implementing custom classifiers."""

    def __init_subclass__(cls: Type[Self], classes: Type[C], **kwargs):
        super().__init_subclass__(**kwargs)
        cls.classes = classes
        cls.check()
        cls.classifiers[cls.CATEGORY] = cls

    @classmethod
    def check(cls):
        """Ensures that the supplied constants are sound."""

        if not any(c for c in cls.classes if c.name.startswith(cls.CATEGORY)):
            raise ValueError(
                f"{cls.classes} does not contain any classes with the prefix {cls.CATEGORY}"
            )
        if cls.CATEGORY.lower() not in cls.VARS:
            raise ValueError(f"VARS does not contain the key {cls.CATEGORY.lower()}")

    @classmethod
    def classify_all(cls, docs: list[Document]) -> dict[str, list[C]]:
        """Runs a Document through all registered subclasses."""

        return {c: klass().run(docs) for c, klass in cls.classifiers.items()}

    def example_template(self, dynamic=set()) -> str:
        """The template used to construct one example in the prompt."""

        return "\n".join(
            [
                f"{v}: { '' if k in dynamic else '{' + k + '}' }"
                for k, v in self.VARS.items()
            ]
        )

    def prompt_template(self):
        """The template used to construct the prompt."""

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
                {self.PREFIX}
                Return a comma-separated list of classes, with no extra text of explanation.
                For example: "industry_software, role_ic"

                Options:
                {classes}

                {self.SUFFIX}

                """
            ),
            suffix=self.example_template(dynamic={self.CATEGORY.lower()}),
            input_variables=list(self.VARS.keys() - {self.CATEGORY.lower()}),
            example_separator="\n",
        )

    def debug_prompt(self, **kwargs: dict[str, str]) -> str:
        """Returns the prompt with the given variables filled in."""

        return self.prompt_template().format(**kwargs)

    def _parse(self, results: str) -> list[C]:
        return [
            c for result in results.split(",") for c in [self.classes.get(result)] if c
        ]

    def run(self, docs: list[Document]) -> list[C]:
        """Runs a Document through the classifier and returns the tags."""
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template())
        results = self.cached(
            "run",
            chain,
            docs,
            self.classify,
        )
        return self._parse(results)

    @abstractmethod
    def classify(self, docs: list[Document]) -> dict[str, str]:
        """Extracts a set of VARS from a list of Documents.
        This method must be implemented by subclasses.

        Args:
            docs: The Documents resulting from apply the Splitter to an import source.

        Returns:
            A dictionary mapping variable names to values. The keys should be a subset of the keys in `VARS`.
        """
        raise NotImplementedError
