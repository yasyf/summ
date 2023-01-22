from operator import attrgetter
from typing import Callable, Union, cast, get_type_hints

from langchain.chains import TransformChain
from langchain.chains.base import Chain as LChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

from user_interview_summary.cache.cacher import ChainCacheItem

TExtract = Callable[[Document], Union[str, dict[str, str]]]


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

    def cached(
        self,
        name: str,
        chain: LChain,
        doc: Document,
        extract: TExtract = attrgetter("page_content"),
    ):
        item = ChainCacheItem(
            klass=self.__class__.__name__,
            name=name,
            document=doc,
        )
        if cached := cast(ChainCacheItem, ChainCacheItem.get(item.pk)):
            return cached.result
        else:
            if isinstance((args := extract(doc)), str):
                item.result = chain.run(args)
            else:
                item.result = chain.run(**args)
            item.save()
            return item.result
