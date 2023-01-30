import json
import logging
import random
import sqlite3
from functools import cached_property
from typing import cast

from langchain import LLMChain, PromptTemplate
from langchain.docstore.document import Document

from summ.shared.utils import dedent
from summ.structure.structurer import Structurer, TVal_


class SQLStructurer(Structurer):
    """Constructs an in-memory SQLite database to store and query the structured data."""

    def __init__(self, query: str, **kwargs):
        super().__init__(query, **kwargs)
        self.llm.max_tokens = 1024
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def metrics_template(self) -> PromptTemplate:
        """The template to transform a query into a list of metrics."""

        return PromptTemplate(
            template=dedent(
                f"""
                Use the query to determine which structured data is needed, and use this to create a SQL table DDL.
                Include a confidence score column with values from 0 to 100.
                If the query is qualitative, you can return an empty table.
                Your response must be valid and complete SQL.

                Prompt: {{{{ query }}}}
                DDL:
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
                You will be provided with the schema for a SQL table.
                Write between zero and three SQL statements which will insert data into the table.
                You do not need to use all three statements.
                Do not insert data which is not relevant to the query. Do not insert data which is ambiguous. Do not insert data which is noisy or too long.
                For each row, record your confidence that the data is relevant to the query as a number from 0 to 100, using the confidence score column.
                Your response must be valid and complete SQL.

                Query: {{ query }}
                Document:
                ```
                {{ text }}
                ```
                Schema:
                ```
                {{ schema }}
                ```
                Response:
                ```
                """
            ),
            input_variables=["query", "text", "schema"],
            template_format="jinja2",
        )

    def clean_template(self) -> PromptTemplate:
        """The template to extract the relevant rows."""

        return PromptTemplate(
            template=dedent(
                """
                Write a SQLite statement which will clean and extract rows from the table.
                Use CTEs to process and clean the data. Apply a WHERE clause to extract the relevant rows.
                The cleaning rules must be short and simple. They do not need to be comprehensive.
                Your response must be valid and complete SQLite.
                No string literal in the response may be longer than 200 characters. The response must be less than 512 tokens.
                The sample data provided is not comprehensive.

                --
                This is an example. Use it as a guide, but do not copy it.
                The content is not relevant to the task.
                Your query can take a different form.
                ```
                WITH CleanedData AS (
                SELECT
                    CASE
                    WHEN Department LIKE '%eng%' THEN 'Engineering'
                    WHEN Department = 'sales' THEN 'Sales & Marketing'
                    WHEN Department = 'marketing' THEN 'Sales & Marketing'
                    ELSE NULL
                    END AS Department,
                    CASE
                    WHEN (Answer = 'yup' OR Answer = 'yes') THEN 'Yes'
                    WHEN Answer LIKE 'no%' THEN 'No'
                    ELSE Answer
                    END AS Answer,
                    Answer
                FROM Table
                )
                SELECT
                Department,
                GROUP_CONCAT(Response, ', ') AS Responses
                FROM CleanedData
                WHERE ConfidenceScore > 80
                GROUP BY Department
                HAVING COUNT(Response) > 2
                ORDER BY AVG(ConfidenceScore) DESC
                LIMIT 10;
                ```
                --

                Now with the real input.

                Query: {{ query }}
                Schema:
                ```
                {{ schema }}
                ```
                Sample Data:
                ```
                {{ data }}
                ```
                Response:
                ```
                """
            ),
            input_variables=["query", "schema", "data"],
            template_format="jinja2",
        )

    @cached_property
    def schema(self) -> str:
        """Extract the DDL for a table from the query."""

        chain = LLMChain(llm=self.llm, prompt=self.metrics_template())
        results = self.cached(
            "sql",
            chain,
            Document(page_content=self.query),
            lambda d: {"query": d.page_content, "stop": "```"},
        )

        return results

    @cached_property
    def table_name(self) -> str:
        curr = self.conn.cursor()
        curr.execute("SELECT * FROM sqlite_schema WHERE type='table';")
        res = curr.fetchone()
        return res["name"]

    def extract_metrics(self, doc: Document):
        """Extract the metrics from the document"""
        self.dprint(f"Metrics for: {doc.metadata['file']}", color="green")
        results = self.cached(
            "extract_metrics",
            LLMChain(llm=self.llm, prompt=self.doc_template()),
            doc,
            lambda d: {
                "query": self.query,
                "text": d.page_content,
                "schema": self.schema,
                "stop": "```",
            },
        )
        self.dprint(results)
        self.dprint.flush("green")
        return results

    def clean(self, metrics: list[sqlite3.Row]) -> list[dict]:
        """Extract cleaned metrics from all documents"""
        stmnt = self.cached(
            "clean_metrics",
            LLMChain(llm=self.llm, prompt=self.clean_template()),
            cast(list[Document], []),
            lambda _: {
                "query": self.query,
                "schema": self.schema,
                "data": "\n".join([str(tuple(dict(x).values())) for x in metrics]),
                "stop": "```",
            },
        )

        try:
            return [dict(x) for x in self.conn.cursor().execute(stmnt).fetchall()]
        except Exception as e:
            logging.info(e)
            return [dict(x) for x in metrics]

    def _extract(self, docs: list[Document]) -> dict[str, TVal_]:
        self.conn.cursor().executescript(self.schema)

        metrics = self._pmap(self.extract_metrics, docs)
        for metric in metrics:
            try:
                self.conn.cursor().executescript(metric)
            except Exception as e:
                logging.info(e)

        metrics = (
            self.conn.cursor()
            .execute(f"SELECT * FROM {self.table_name} ORDER BY RANDOM() LIMIT 50;")
            .fetchall()
        )
        return {"data": self.clean(metrics)}
