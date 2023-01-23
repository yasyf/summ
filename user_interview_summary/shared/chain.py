import logging
from operator import attrgetter
from typing import Callable, TypeVar, Union, cast, get_type_hints, overload

from langchain.chains import TransformChain
from langchain.chains.base import Chain as LChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

from user_interview_summary.cache.cacher import CacheDocument, ChainCacheItem

TDoc = TypeVar("TDoc", bound=Union[Document, list[Document]])
TExtract = Callable[[TDoc], Union[str, dict[str, str], TDoc]]


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

    @overload
    def cached(
        self,
        name: str,
        chain: LChain,
        doc: Document,
        extract: TExtract[Document] = attrgetter("page_content"),
    ) -> str:
        ...

    @overload
    def cached(
        self,
        name: str,
        chain: LChain,
        doc: list[Document],
        extract: TExtract[list[Document]],
    ) -> str:
        ...

    def cached(
        self,
        name: str,
        chain: LChain,
        doc: TDoc,
        extract: TExtract[TDoc] = cast(TExtract[Document], attrgetter("page_content")),
    ):
        item = ChainCacheItem.passthrough(
            klass=self.__class__.__name__,
            name=name,
            document=[CacheDocument.from_doc(d) for d in doc]
            if isinstance(doc, list)
            else CacheDocument.from_doc(doc),
        )

        if item.result:
            logging.info(f"Cache hit for {item.pk}")
            return item.result
        else:
            logging.warning(f"Cache miss for {item.pk}")
            if not isinstance((args := extract(doc)), dict):
                item.result = chain.run(args)  # type: ignore
            else:
                item.result = chain.run(**args)
            item.save()
            return item.result
