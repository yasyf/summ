import itertools
import logging
import os
import re
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from operator import attrgetter
from threading import RLock, current_thread
from typing import (
    Any,
    Callable,
    ClassVar,
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
from pydantic import BaseModel
from retry import retry
from termcolor import colored

from summ.cache.cacher import CacheDocument, ChainCacheItem

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
R = TypeVar("R")

TDoc = TypeVar("TDoc", bound=Union[Document, list[Document]])
TExtract = Callable[[TDoc], Union[str, TDoc, dict[str, str], dict[str, TDoc]]]

audit_lock = RLock()


def locked(lock: RLock):
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class Entry(BaseModel):
    thread: int
    color: Optional[str]
    indent: int
    title: str
    text: str
    parent: Optional["Entry"]

    class Config:
        frozen = True


class DPrinter:
    """A thread-safe pretty-printer for debug use."""

    Entry = Entry

    main_thread: ClassVar[int] = 0
    main_indents: ClassVar[dict[Type["Chain"], int]] = defaultdict(int)
    instances: ClassVar[ContextVar[dict[Type["Chain"], Self]]] = ContextVar("instances")
    auditors: ClassVar[list[Callable[[Entry], None]]] = list()

    last_entry: Optional[Entry] = None

    @classmethod
    def get(cls, instance: Type["Chain"], *args, **kwargs) -> Self:
        if not cls.main_thread:
            cls.main_thread = current_thread().native_id
        instances = cls.instances.get({})
        if instance not in instances:
            instances[instance] = cls(instance, *args, **kwargs)
            cls.instances.set(instances)
        return instances[instance]

    @classmethod
    def register_auditor(cls, auditor: Callable[[Entry], None]):
        with audit_lock:
            cls.auditors.append(auditor)

        @locked(audit_lock)
        def unregister():
            cls.auditors.remove(auditor)

        return unregister

    def audit(self, color: Optional[str], title: str, text: str):
        entry = self.last_entry = self.Entry(
            color=color,
            indent=self._indent,
            title=title,
            text=text,
            thread=current_thread().native_id,
            parent=self.last_entry.copy(update={"parent": None})
            if self.last_entry
            else None,
        )
        with audit_lock:
            for auditor in self.auditors:
                auditor(entry)

    def __init__(self, instance: Type["Chain"], debug: bool = False):
        self.instance = instance
        self.debug = debug
        self.__indent = 0
        self._outputs: dict[int, dict[int, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

    @contextmanager
    def indent_children(self):
        self.indent()
        try:
            yield
        finally:
            self.dedent()

    def indent(self):
        self._indent += 1

    def dedent(self):
        self._indent -= 1
        self._flush()

    def reset(self):
        self._indent = 0

    @property
    def _indent(self) -> int:
        if self._is_main:
            return self.__class__.main_indents[self.instance]
        else:
            return self.__class__.main_indents[self.instance] + self.__indent

    @_indent.setter
    def _indent(self, val: int):
        if self._is_main:
            self.__class__.main_indents[self.instance] = val
        else:
            self.__indent = val - self.__class__.main_indents[self.instance]

    @property
    def _is_main(self):
        return current_thread().native_id == self.main_thread

    def _flush(self):
        if not self._is_main:
            return
        for tid, outputs in self._outputs.items():
            for indent, strings in sorted(outputs.items()):
                self._print(strings)
                self._outputs[tid][indent] = list()

    def _print(self, strings: list[str]):
        for s in strings:
            if self.debug:
                print(s)
            else:
                logging.debug(s)

    def _append(self, s: str):
        if self._is_main:
            self._print([s])
        else:
            self._outputs[current_thread().native_id][self._indent].append(s)

    def _pprint(self, obj: Union[list[dict[str, T]], dict[str, T]]):
        if isinstance(obj, list):
            return "\n⎯\n".join([self._pprint(o) for o in obj])
        elif hasattr(obj, "dict"):
            obj = obj.dict()
            return "\n".join(
                [
                    colored(k, attrs=["bold"]) + ": " + str(v)
                    for k, v in obj.items()
                    if v
                ]
            )
        return str(obj)

    def __call__(
        self,
        s: Union[str, dict, list],
        obj: Optional[Any] = None,
        color: Optional[str] = None,
    ):
        if not isinstance(s, str):
            s = self._pprint(s)

        if obj:
            pprinted = self._pprint(obj)
            self.audit(color, s, pprinted)
            s = f"{s}: {pprinted}"
        else:
            self.audit(color, "", s)

        indent_ = "\n" + ("  " * self._indent)
        try:
            width = os.get_terminal_size().columns - 30
        except OSError:
            width = 80
        formatted = indent_ + colored(
            indent_.join(
                itertools.chain.from_iterable(
                    [textwrap.wrap(l, width=width) for l in s.splitlines()]
                )
            ),
            color=color,
            attrs=["bold"] if color else [],
        )

        if self._indent:
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

    def _pprogress(self):
        try:
            if self.pool._original_iterator is None:
                return (self.pool.n_completed_tasks, self.pool.n_dispatched_tasks)
            else:
                return (self.pool.n_completed_tasks, None)
        except AttributeError:
            return (0, None)

    def _ppprogress(self):
        done, total = self._pprogress()
        return f"[{done}/{total}]" if total else f"[?/{done}]"

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
