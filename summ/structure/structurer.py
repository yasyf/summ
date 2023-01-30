import json
import logging
from collections import Counter
from enum import StrEnum
from functools import cached_property
from typing import Generic, Iterable, Optional, TypeVar, Union, cast

import dirtyjson
from langchain import LLMChain, PromptTemplate
from langchain.docstore.document import Document

from summ.classify.classes import Classes
from summ.shared.chain import Chain
from summ.shared.models import WithSafeParse
from summ.shared.utils import dedent

TVal_ = Union[str, int, float, list, dict]
TVal = TypeVar("TVal", bound=TVal_)


class MetricType(Classes, StrEnum):
    ENUM = "enum"
    STRING = "string"
    NUMBER = "number"
    LIST = "list"


class MetricCollect(Classes, StrEnum):
    LIST = "list"
    SUM = "sum"
    COUNT = "count"
    COUNT_UNIQUE = "count_unique"
    AVERAGE = "average"


class Metric(WithSafeParse):
    metric: str
    prompt: str
    type: MetricType
    collect: MetricCollect
    options: Optional[list[str]] = None

    def collect_fn(self, values: Iterable["MetricValue[TVal]"]):
        if self.collect == MetricCollect.LIST and self.type == MetricType.STRING:
            return [x.value for x in values]
        elif self.collect == MetricCollect.SUM and self.type == MetricType.NUMBER:
            return sum([cast(int | float, x.value) for x in values])
        elif self.collect == MetricCollect.AVERAGE and self.type == MetricType.NUMBER:
            return sum([cast(int | float, x.value) for x in values]) / len(list(values))
        elif self.collect == MetricCollect.COUNT:
            return len(list(values))
        elif self.collect == MetricCollect.COUNT_UNIQUE:
            return dict(Counter([x.value for x in values]).most_common())
        else:
            raise RuntimeError(
                f"Unknown collect method {self.collect} for type {self.type}"
            )


class MetricValue(Generic[TVal], WithSafeParse):
    metric: Metric
    value: TVal


class Structurer(Chain):
    """Structurers infer a set of structured data to extract based on the query, then extract that data from every source document."""

    def __init__(self, query: str, **kwargs):
        super().__init__(**kwargs)
        self.query = query

    def metrics_template(self) -> PromptTemplate:
        """The template to transform a query into a list of metrics."""
        return PromptTemplate(
            template=dedent(
                f"""
                Use the query to determine which structured data is needed, and for each, write a specification which will extract and collect the data.
                If the query is qualitative, you can return an empty list.
                Your response must be in valid JSON format. Do not extract the information yet, just describe how to do so.
                The options for type are: {', '.join(list(MetricType))}.
                The options for collect are: {', '.join(list(MetricCollect))}.
                The prompt should minimize variance in the response.

                For example:
                Prompt: In each department, how many times did people prefer Google over Bing.
                Response:
                ```
                [
                    {{"metric": "department", "prompt": "Extract the company department that the user of this interview works in.", "type": "string", "collect": "list"}},
                    {{"metric": "preferred", "prompt": "Which of the following options best represents which search engine was preferred?", "type": "enum", "options": ["GOOGLE", "BING", "OTHER"], "collect": "count_unique"}},
                ]
                ```

                Prompt: {{{{ query }}}}
                Response:
                ```
                """
            ),
            input_variables=["query"],
            template_format="jinja2",
        )

    def doc_template(self) -> PromptTemplate:
        """The template to transform a document into a list of metrics."""

        return PromptTemplate(
            template=dedent(
                """
                Your task is to take a spec describing how to extract structured data, and apply it to a document.
                You must follow the spec exactly. For example, if the spec specifies an enum, your response must be one of the options.

                For example:
                Document:
                ```
                Yea over in sales, we prefer Google. Rest of the company likes DuckDuckGo.
                ```
                Spec:
                ```
                [
                    {"metric": "department", "prompt": "Extract the company department that the user of this interview works in.", "type": "string", "collect": "list"},
                    {"metric": "preferred", "prompt": "Which of the following options best represents which search engine was preferred?", "type": "enum", "options": ["GOOGLE", "BING", "OTHER"], "collect": "count_unique"},
                ]
                ```
                Response:
                ```
                {"department": "Engineering", "preferred": "GOOGLE"}
                ```

                Document:
                ```
                {{ text }}
                ```
                Spec:
                ```
                {{ spec }}
                ```
                Response:
                ```
                """
            ),
            input_variables=["text", "spec"],
            template_format="jinja2",
        )

    def clean_template(self) -> PromptTemplate:
        """The template to take a set of extracted metrics and clean them up."""

        return PromptTemplate(
            template=dedent(
                """
                Your task is to take a set of extracted metrics and clean them up.
                Each metric will have a spec. You can use the spec to determine how to clean the metric.
                The goal is to minimize variance and make this data useful for aggregation.
                If elements are semantically equivalent, they should be combined.

                For example:
                Query: In each department, how many times did people prefer Google over Bing.
                Spec:
                ```
                [
                    {"metric": "department", "prompt": "Extract the company department that the user of this interview works in.", "type": "string", "collect": "list"},
                    {"metric": "preferred", "prompt": "Which of the following options best represents which search engine was preferred?", "type": "enum", "options": ["GOOGLE", "BING", "OTHER"], "collect": "count_unique"},
                    {"metric": "feelings", "prompt": "How did the individual feel about the search engine?", "type": "string", "collect": "list"},
                ]
                ```
                Data:
                ```
                [
                    {
                        "department": ["Engineering", "eng", "engineering", "sales", "marketing", "markting"],
                        "preferred": {"GOOGLE": 3, "BING": 1, "OTHER": 1}
                        "feelings": ["it was truly awesome", "I really liked it", "awesome for all the months I used it", "sweet and liked it quite a bit"]
                    },
                ]
                ```
                Cleaned:
                ```
                [
                    {
                        "department": ["Engineering", "Sales", "Marketing"],
                        "preferred": {"GOOGLE": 3, "BING": 1}
                        "feelings": ["Awesome", "Liked It", "Sweet"],
                    },
                ]
                ```

                Spec:
                ```
                {{ spec }}
                ```
                Data:
                ```
                {{ data }}
                ```
                Cleaned:
                ```
                """
            ),
            input_variables=["spec", "data"],
            template_format="jinja2",
        )

    @cached_property
    def metrics(self) -> list[Metric]:
        """Extract the metrics from the query"""
        chain = LLMChain(llm=self.llm, prompt=self.metrics_template())
        results = self.cached(
            "metrics",
            chain,
            Document(page_content=self.query),
            lambda d: {"query": d.page_content, "stop": "```"},
        )
        try:
            return [
                m
                for o in cast(list, dirtyjson.loads(results))
                for m in [Metric.safe_parse(o)]
                if m is not None
            ]
        except Exception as e:
            logging.info(e)
            return []

    def metric(self, metric: str) -> Metric:
        return next(m for m in self.metrics if m.metric == metric)

    @property
    def spec(self) -> str:
        return f"[{', '.join([m.json() for m in self.metrics])}]"

    def extract_metrics(self, doc: Document) -> dict[str, MetricValue]:
        """Extract the metrics from the document"""
        self.dprint(f"Metrics for: {doc.metadata['file']}", color="green")
        results = self.cached(
            "extract_metrics",
            LLMChain(llm=self.llm, prompt=self.doc_template()),
            doc,
            lambda d: {
                "text": d.page_content,
                "spec": self.spec,
                "stop": "```",
            },
        )
        try:
            metrics = {
                k: m
                for k, v in cast(dict[str, TVal], dirtyjson.loads(results)).items()
                for m in [
                    MetricValue.safe_parse({"metric": self.metric(k), "value": v})
                ]
                if m and m.value is not None
            }
        except Exception as e:
            logging.info(e)
            metrics = {}

        self.dprint(
            [{"metric": m.metric.metric, "value": m.value} for m in metrics.values()],
        )
        self.dprint.flush("green")
        return metrics

    def clean(self, metrics: dict[str, TVal_]) -> dict[str, TVal_]:
        """Clean up a set of extracted metrics."""

        results = self.cached(
            "clean_metrics",
            LLMChain(llm=self.llm, prompt=self.clean_template()),
            cast(list[Document], []),
            lambda _: {
                "spec": self.spec,
                "data": json.dumps(metrics),
                "stop": "```",
            },
        )

        try:
            return dirtyjson.loads(results)
        except Exception as e:
            logging.info(e)
            return metrics

    def _extract(self, docs: list[Document]) -> dict[str, TVal_]:
        metrics = self._pmap(self.extract_metrics, docs)
        formatted: dict[str, TVal_] = {
            m.metric: m.collect_fn(cs)
            for m in self.metrics
            for cs in [[x for v in metrics for x in [v.get(m.metric, None)] if x]]
        }
        return self.clean(formatted)

    def extract(self, docs: list[Document]) -> dict[str, TVal_]:
        """Extract metrics from all documents"""
        try:
            return self._extract(docs)
        except Exception as e:
            logging.info(e)
            return {}
