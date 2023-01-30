import itertools
import logging
import os
import re
import textwrap
from collections import defaultdict
from operator import attrgetter
from threading import RLock, current_thread, local, main_thread
from typing import (
    Callable,
    Iterable,
    Optional,
    Self,
    Type,
    TypeVar,
    TypeVarTuple,
    Union,
    cast,
    get_type_hints,
    overload,
)

from joblib import Parallel, delayed
from langchain.chains import TransformChain
from langchain.chains.base import Chain as LChain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from openai.error import RateLimitError
from retry import retry
from termcolor import colored

from summ.cache.cacher import CacheDocument, ChainCacheItem

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
R = TypeVar("R")

TDoc = TypeVar("TDoc", bound=Union[Document, list[Document]])
TExtract = Callable[[TDoc], Union[str, TDoc, dict[str, str], dict[str, TDoc]]]

thread_local = local()
print_lock = RLock()


class DPrinter:
    """A thread-safe pretty-printer for debug use."""

    main_indents = defaultdict(int)

    @classmethod
    def get(cls, instance, *args, **kwargs) -> Self:
        if not hasattr(thread_local, "dprinters"):
            thread_local.dprinters = {}
        if instance not in thread_local.dprinters:
            thread_local.dprinters[instance] = cls(instance, *args, **kwargs)
        return thread_local.dprinters[instance]

    def __init__(self, instance, debug: bool = False):
        self.instance = instance
        self.debug = debug
        self._parents = []
        self.__indent = 0
        self._indents = set()
        self._outputs = defaultdict(list)

    def indent(self):
        self._indent += 1

    def dedent(self):
        self._indent -= 1

    def reset(self):
        self._indent = 0

    @property
    def _parent(self) -> str:
        return self._parents[-1]

    @_parent.setter
    def _parent(self, color: str):
        self._parents.append(color)

    @property
    def _indent(self) -> int:
        if main_thread() == current_thread():
            return self.__class__.main_indents[self.instance]
        else:
            return self.__class__.main_indents[self.instance] + self.__indent

    @_indent.setter
    def _indent(self, val: int):
        if main_thread() == current_thread():
            self.__class__.main_indents[self.instance] = val
        else:
            self.__indent = val

    def _flush(self, color: str):
        assert self._parent == color
        self._print(self._outputs[color])
        self.dedent()
        self._parents.pop()
        self._indents.remove(color)
        self._outputs[color] = []

    def flush(self, color: str):
        while self._parent != color:
            self._flush(self._parent)
        self._flush(color)

    def _print(self, strings: list[str]):
        with print_lock:
            for s in strings:
                if self.debug:
                    print(s)
                else:
                    logging.debug(s)

    def _append(self, s: str):
        if main_thread() == current_thread():
            self._print([s])
        else:
            self._outputs[self._parent].append(s)

    def _pprint(self, obj: Union[list[dict[str, T]], dict[str, T]]):
        if isinstance(obj, list):
            return "\n⎯\n".join([self._pprint(o) for o in obj])
        if hasattr(obj, "dict"):
            obj = obj.dict()
        return "\n".join(
            [colored(k, attrs=["bold"]) + ": " + str(v) for k, v in obj.items() if v]
        )

    def __call__(
        self,
        s: Union[str, dict, list],
        color: Optional[str] = None,
        indent: bool = True,
        dedent: bool = True,
    ):
        if not isinstance(s, str):
            s = self._pprint(s)

        if color and color in self._indents and dedent:
            self.flush(color)

        indent_ = "\n" + ("  " * self._indent)
        formatted = indent_ + colored(
            indent_.join(
                itertools.chain.from_iterable(
                    [
                        textwrap.wrap(l, width=os.get_terminal_size().columns - 30)
                        for l in s.splitlines()
                    ]
                )
            ),
            color=color,
            attrs=["bold"] if color else [],
        )

        if color and color not in self._indents and indent:
            self._parent = color
            self.indent()
            self._indents.add(color)
            self._append(formatted)
        elif self._indent:
            self._append(formatted)
        else:
            self._print([formatted])


class Chain:
    """The base class of most operations.

    Provides shared facilities for querying LLMs, parsing response, and caching.
    """

    def __init__(self, debug: bool = False, verbose: bool = False):
        self.llm = OpenAI(temperature=0.0)
        self.pool = Parallel(n_jobs=-1, prefer="threads", verbose=10 if verbose else 0)
        self.verbose = verbose
        self.debug = debug

    def spawn(self, cls: Type[T], **kwargs) -> T:
        instance = cls(debug=self.debug, verbose=self.verbose, **kwargs)
        instance.dprint._indent = self.dprint._indent
        return instance

    @property
    def dprint(self):
        return DPrinter.get(instance=self.__class__, debug=self.debug)

    def _parse(self, results: list[str], prefix: str = ""):
        return [
            g.strip()
            for r in results
            for p in [re.search(prefix + r"(?:\s*)(?P<res>.*)", r)]
            for g in [p and p.group("res")]
            if p and g
        ]

    def _pmap(
        self, meth: Callable[[T, *Ts], R], it: Iterable[T], *args: *Ts
    ) -> list[R]:
        return self._parallel(meth, [(x, *args) for x in it])

    def _parallel(self, meth: Callable[[*Ts], R], it: Iterable[tuple[*Ts]]) -> list[R]:
        return self.pool(delayed(meth)(*x) for x in it) or []

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

    @retry(exceptions=RateLimitError, tries=5, delay=6, jitter=(0, 4))
    def _run_with_retry(self, chain: LChain, *args, **kwargs):
        return chain.run(*args, **kwargs)

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
        """Caches the result of a [langchain `Chain`][langchain.chains.LLMChain].

        Args:
            name: The name of the function calling the cache.
            chain (langchain.chains.LLMChain): The Chain to run.
            doc: The document to run the chain on.
            extract: A function to extract the arguments from the document.
        """
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
                item.result = self._run_with_retry(chain, args)  # type: ignore
            else:
                item.result = self._run_with_retry(chain, **args)
            item.save()
            return item.result
