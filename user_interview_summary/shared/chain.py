from typing import Callable, get_type_hints

from langchain.chains import TransformChain
from langchain.llms import OpenAI


class Chain:
    def __init__(self):
        self.llm = OpenAI(temperature=0.0)

    @classmethod
    def to_chain(cls, method: str) -> TransformChain:
        meth: Callable = getattr(cls(), method)
        hints = get_type_hints(meth)

        def transform_func(inputs: dict) -> dict:
            return {"output": meth(**inputs)}

        return TransformChain(
            input_variables=list(hints.keys()),
            output_variables=["output"],
            transform=transform_func,
        )
