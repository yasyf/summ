import itertools
from typing import TypedDict, cast, overload

import pinecone
from langchain import (
    BasePromptTemplate,
    FewShotPromptTemplate,
    LLMChain,
    PromptTemplate,
)
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

from summ.classify.classes import Classes
from summ.embed.embedder import Embedding
from summ.shared.chain import Chain
from summ.shared.utils import dedent
from summ.summarize.summarizer import Summarizer


class Fact(TypedDict):
    fact: str
    context: str
    attributes: str


class Answer(TypedDict):
    question: str
    answer: str


class Conclusion(TypedDict):
    step: str
    conclusion: str


class Querier(Chain):
    """Queriers are responsible for answering questions about the dataset,
    using the pre-populated model.

    The high level flow is as follows:
    1. Determine a set of sub-questions necessary to answer the original.
    2. For each sub-question, determine a set of queries that would render relevant facts.
    3. For each query, search the vector store for facts _or_ queries that are similar to the embedded query.
    4. Extract the facts from these results.
    5. Recursively summarize up the tree until the original question is answered.
    """

    FACT_PROMPT = PromptTemplate(
        input_variables=["fact", "context", "attributes"],
        template=dedent(
            """
                The user was tagged as: {attributes}.
                Their respoonse was: {fact}
                A summary of the whole interview is: {context}
            """
        ),
    )

    def __init__(self, index: str, debug: bool = False):
        super().__init__(debug=debug)
        self.index_name = index
        self.embeddings = OpenAIEmbeddings()
        self.summarizer = Summarizer()
        self.index = pinecone.Index(index)
        self.facts = set()

    # Questions

    def steps_template(self):
        """The template to determine sub-questions."""

        return PromptTemplate(
            template=dedent(
                """
                Your task is to determine a set of up to {n} steps which would answer a question.
                You must not answer the question, merely determine the best way to answer it.

                For example, if the question was "What is the most popular house colors?":
                1. Determine all the possible colors that houses can be.
                2. Determine the number of houses that are each color.
                3. Determine the most popular color.

                The question is: {query}

                1.
                """
            ),
            input_variables=["query", "n"],
        )

    def queries_template(self):
        """The template to determine sub-question queries."""

        return PromptTemplate(
            template=dedent(
                """
                Your task is to determine a set of queries which would answer a question.
                The queries run against a database of facts compiled across several user interviews.

                The overall question you are trying to answer is: {query}
                You are on the following step: {step}

                Generate a bulleted list of up to {n} natural-language queries to complete this step.

                -
                """
            ),
            input_variables=["query", "step", "n"],
        )

    # Answers

    def facts_template(self, facts: list[Fact]):
        """The template to summarize sub-question queries."""

        return FewShotPromptTemplate(
            examples=cast(list[dict], facts),
            example_prompt=self.FACT_PROMPT,
            prefix=dedent(
                """
                Your task is to answer a query against a corpus of user interviews.
                To help answer the question, you are provided with a set of facts (along with the context and attributes of the author of the fact).

                The query is: {query}

                The relevant facts are:
                """
            ),
            suffix="Your response:\n",
            input_variables=["query"],
            example_separator="\n",
        )

    def answers_template(self, answers: list[Answer]):
        """The template to summarize sub-questions."""

        return FewShotPromptTemplate(
            examples=cast(list[dict], answers),
            example_prompt=PromptTemplate(
                template=dedent(
                    """
                    Query:
                    {question}

                    Answer:
                    {answer}
                """
                ),
                input_variables=["question", "answer"],
            ),
            prefix=dedent(
                """
                Your task is to take a set of queries and answers, and use them to complete a step towards answering an original question.

                The original question you are trying to answer is: {query}
                You are on the following step: {step}

                Here are the queries and answers:
                """
            ),
            suffix="Completed step:\n",
            input_variables=["query", "step"],
        )

    def conclusions_template(self, conclusions: list[Conclusion]):
        """The template to summarize the final answer."""

        return FewShotPromptTemplate(
            examples=cast(list[dict], conclusions),
            example_prompt=PromptTemplate(
                template=dedent(
                    """
                    Step:
                    {step}

                    Conclusion:
                    {conclusion}
                """
                ),
                input_variables=["step", "conclusion"],
            ),
            prefix=dedent(
                """
                Your task is to take a set of steps that were conducted to answer a question, and use them to answer that question.

                The question you are trying to answer: {query}

                The steps you went through to answer this question are:
                """
            ),
            suffix="Final answer:\n",
            input_variables=["query"],
        )

    @overload
    def _query(self, prompt: BasePromptTemplate, **kwargs) -> str:
        ...

    @overload
    def _query(
        self, prompt: BasePromptTemplate, initial: str, prefix: str, **kwargs
    ) -> list[str]:
        ...

    def _query(
        self, prompt: BasePromptTemplate, initial: str = "", prefix: str = "", **kwargs
    ):
        chain = LLMChain(llm=self.llm, prompt=prompt)
        results = initial + chain.run(**kwargs)
        self.dprint(results)
        if initial and prefix:
            return self._parse(results.splitlines(), prefix)
        else:
            return results

    def _query_facts(self, query: str, n: int, classes: list[Classes]):
        embedding = self.embeddings.embed_query(query)
        filter = {"$or": [{"classes": c.value} for c in classes]} if classes else None
        results = self.index.query(
            embedding, top_k=n * 3, filter=filter  # type: ignore
        )["matches"]

        facts: list[Fact] = [
            {
                "fact": e.fact,
                "context": e.document.metadata["summary"],
                "attributes": ", ".join(
                    itertools.chain.from_iterable(
                        e.document.metadata["classes"].values()
                    )
                ),
            }
            for r in results
            for e in [Embedding.safe_get(r["id"])]
            if e
        ]

        new_facts = {f["fact"]: f for f in facts if f["fact"] not in self.facts}
        old_facts = {f["fact"]: f for f in facts if f["fact"] in self.facts}
        facts = (list(new_facts.values()) + list(old_facts.values()))[:n]

        self.facts.update(f["fact"] for f in facts)

        if not facts:
            raise RuntimeError("No vectors found!")

        return facts

    def _answer_question(self, question: str, n: int, classes: list[Classes]) -> Answer:
        self.dprint(f"Answer to: {question}", color="magenta")
        facts = self._query_facts(question, n, classes)
        answer = self.summarizer.summarize_facts(
            question,
            [Document(page_content=self.FACT_PROMPT.format(**f)) for f in facts],
        )
        self.dprint(answer)
        self.dprint.flush("magenta")
        return {"question": question, "answer": answer}

    def _conclude_step(
        self, step: str, query: str, n: int, classes: list[Classes]
    ) -> Conclusion:
        self.dprint(f"Questions for: {step}", color="cyan")
        questions = self._query(
            self.queries_template(), "-", r"\-", query=query, step=step, n=n
        )
        answers = self._pmap(self._answer_question, questions, n, classes)
        self.dprint(f"Solved: {step}", color="cyan")
        conclusion = self._query(self.answers_template(answers), query=query, step=step)
        return {"step": step, "conclusion": conclusion}

    def query(self, query: str, n: int = 3, classes: list[Classes] = []):
        """Runs the entire question-answering process.

        Args:
            query: The question to ask.
            n: The number of facts to use from the vector store per query.
            classes: The interview tags to use as filters (AND).

        Returns:
            The answer to the question.
        """
        self.dprint(f"Steps for: {query}", color="green")
        steps = self._query(self.steps_template(), "1.", r"\d+(?:\.)", query=query, n=n)
        conclusions = [self._conclude_step(s, query, n, classes) for s in steps]
        self.dprint(f"Answer: {query}", color="green")
        return self._query(self.conclusions_template(conclusions), query=query)
