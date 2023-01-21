from langchain.llms import OpenAI


class Chain:
    def __init__(self):
        self.llm = OpenAI(temperature=0.0)
