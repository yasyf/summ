from langchain import PromptTemplate

from summ.shared.chain import Chain
from summ.shared.utils import dedent


class Structurer(Chain):
    def structured_template(self):
        """The template to XXX"""

        return PromptTemplate(
            template=dedent(
                """
                Your task is to determine what pieces of structured data, if any, need to be extracted from a set of text documents to answer a question.
                Use the query to determine which structured data is needed, and for each, write a specification which will extract and collect the data.
                Your response must be in valid JSON format. Do not extract the information yet, just describe how to do so.
                The options for type are: enum, string, number, list, and date.
                The options for collect are: list, sum, count, and average.
                Only use an enum if you are confident you can enumerate the entire space. If the type is enum, you must specify a set of options.

                For example:
                Prompt: In each department, how many times did people prefer Google over Bing.
                Response:
                ```
                [
                    {"metric": "department", "prompt": "Extract the company department that the user of this interview works in.", "type": "string", "collect": "list"},
                    {"metric": "preferred", "prompt": "Which of the following options best represents which search engine was preferred?", "type": "enum", "options": ["GOOGLE", "BING", "OTHER"], "collect": "list"},
                ]
                ```

                Prompt: {query}
                """
            ),
            input_variables=["query"],
        )

    def sql_template(self):
        """The template to XXX"""

        return PromptTemplate(
            template=dedent(
                """
                Your task is to determine what pieces of structured data, if any, need to be extracted from a set of text documents to answer a question.
                Use the query to determine which structured data is needed, and for each, write a prompt which will both extract and collect the data.

                For example:
                Prompt: What were the most

                1.
                """
            ),
            input_variables=["query", "n"],
        )
