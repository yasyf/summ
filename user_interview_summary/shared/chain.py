import itertools
import logging
import re
import textwrap
from operator import attrgetter
from typing import Callable, Optional, TypeVar, Union, cast, get_type_hints, overload

from langchain.chains import TransformChain
from langchain.chains.base import Chain as LChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from termcolor import colored

from user_interview_summary.cache.cacher import CacheDocument, ChainCacheItem

TDoc = TypeVar("TDoc", bound=Union[Document, list[Document]])
TExtract = Callable[[TDoc], Union[str, dict[str, Union[str, TDoc]], TDoc]]


class DPrinter:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._indent = 0
        self._indents = set()

    def indent(self):
        self._indent += 1

    def dedent(self):
        self._indent -= 1

    def __call__(
        self,
        s: str,
        color: Optional[str] = None,
        indent: bool = True,
        dedent: bool = True,
    ):
        if color and color in self._indents and dedent:
            self._indents.remove(color)
            self.dedent()

        indent_ = "\n" + ("  " * self._indent)
        formatted = indent_ + colored(
            indent_.join(
                itertools.chain.from_iterable(
                    [textwrap.wrap(l) for l in s.splitlines()]
                )
            ),
            color=color,
            attrs=["bold"] if color else [],
        )
        if self.debug:
            print(formatted)
        else:
            logging.debug(formatted)

        if color and color not in self._indents and indent:
            self.indent()
            self._indents.add(color)


class Chain:
    def __init__(self, debug: bool = False):
        self.llm = OpenAI(temperature=0.0)
        self.dprint = DPrinter(debug=debug)

    def _parse(self, results: list[str], prefix: str = ""):
        return [
            g.strip()
            for r in results
            for p in [re.search(prefix + r"(?:\s*)(?P<res>.*)", r)]
            for g in [p and p.group("res")]
            if p and g
        ]

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
        args = extract(doc)
        meta = (
            {k: v for k, v in args.items() if not isinstance(v, (list, Document))}
            if isinstance(args, dict)
            else {}
        )

        item = ChainCacheItem.passthrough(
            klass=self.__class__.__name__,
            name=name,
            meta=meta,
            document=[CacheDocument.from_doc(d) for d in doc]
            if isinstance(doc, list)
            else CacheDocument.from_doc(doc),
        )

        if item.result:
            logging.info(f"Cache hit for {item.pk}")
            return item.result
        else:
            logging.info(f"Cache miss for {item.pk}")
            if not isinstance(args, dict):
                item.result = chain.run(args)  # type: ignore
            else:
                item.result = chain.run(**args)
            item.save()
            return item.result
