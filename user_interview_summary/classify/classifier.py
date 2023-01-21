from textwrap import dedent

from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

from user_interview_summary.classify.classes import Classes
from user_interview_summary.shared.chain import Chain


class Classifier(Chain):
    # def classify(self, doc: Document) -> list[Classes]:
    def classify(self, doc: Document):
        text, title = doc.page_content, doc.metadata["file"]

        print("Title: ", title)

        # First, create the list of few shot examples.
        examples = [
            {"title": "Brendan - UiPath - Sales", "department": "Sales"},
            {
                "title": "Amy - Blue Prism - Lead Automation Developer",
                "department": "RPA Development",
            },
        ]

        # Next, we specify the template to format the examples we have provided.
        # We use the `PromptTemplate` class for this.
        example_formatter_template = """
        User Interview Title: {title}
        Department Classification: {department}\n
        """
        example_prompt = PromptTemplate(
            input_variables=["title", "department"],
            template=example_formatter_template,
        )

        # Finally, we create the `FewShotPromptTemplate` object.
        self.PROMPT_TEMPLATE = FewShotPromptTemplate(
            # These are the examples we want to insert into the prompt.
            examples=examples,
            # This is how we waånt to format the examples when we insert them into the prompt.
            example_prompt=example_prompt,
            # The prefix is some text that goes before the examples in the prompt.
            # Usually, this consists of intructions.
            prefix=dedent(
                """
            The following is a title of a user interview. The user interviewed works in one or more of the following departments:
            Sales
            Finance
            HR
            IT
            Marketing
            Engineering
            Data Science
            Legal
            Medical
            Operations
            C-Suite
            RPA Center of Excellence
            RPA Developement
            Customer Success

            Return only a list of departments, no explanation. For example: "Operations, Sales" if the user works in both Operations and Sales. The form of the title is "Name - Name of Company - Role". Note if a user works at a company that builds RPA software or is an RPA consultancy, they do not work at the RPA Center of Excellence, it means they are an RPA developer to external customers.
            """
            ),
            # The suffix is some text that goes after the examples in the prompt.
            # Usually, this is where the user input will go
            suffix="User Interview Title: {input}\nDepartment:",
            # The input variables are the variables that the overall prompt expects.
            input_variables=["input"],
            # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
            example_separator="\n",
        )

        # We can now generate a prompt using the `format` method.
        # PROMPT_TEMPLATE = few_shot_prompt.format(
        #         input=title
        #     )

        # The following is a title of a user interview. The user interviewed works in one or more of the following departments:
        # Sales
        # Finance
        # HR
        # IT
        # Marketing
        # Engineering
        # Data Science
        # Legal
        # Medical
        # Operations
        # C-Suite
        # RPA Center of Excellence
        # RPA Developement
        # Customer Success

        # Return only a list of departments, no explanation. For example: "Operations, Sales" if the user works in both Operations and Sales. The form of the title is "Name - Name of Company - Role". Note if a user works at a company that builds RPA software or is an RPA consultancy, they do not work at the RPA Center of Excellence, it means they are an RPA developer to external customers.

        # User Interview Title: Brendan - UiPath - Sales
        # Department Classification: Sales

        # User Interview Title: Amy - Blue Prism - Lead Automation Developer
        # Department Classification: RPA Development

        # User Interview Title: Ricardo - Automation Anywhere - Customer Success
        # Department:

        chain = LLMChain(llm=self.llm, prompt=self.PROMPT_TEMPLATE)
        results = chain.run(title)

        return results
