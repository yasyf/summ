from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from summ.shared.chain import Chain, LChain
from summ.shared.utils import dedent


class Summarizer(Chain):
    """Summarizers are responsible for compressing a list of documents into a single one.

    Depending on the type of document, various best-practice summarization methods are used.
    """

    def _summarize(self, name: str, chain: LChain, docs: list[Document]):
        return self.cached(name, chain, docs, lambda x: x).strip()

    def summarize_class(self, docs: list[Document]):
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return self._summarize("summarize_class", chain, docs)

    def summarize_file(self, docs: list[Document]):
        chain = load_summarize_chain(self.llm, chain_type="refine")
        return self._summarize("summarize_file", chain, docs)

    def summarize_doc(self, doc: Document):
        chain = load_summarize_chain(self.llm, chain_type="stuff")
        return self._summarize("summarize_doc", chain, [doc])

    def summarize_facts(self, query: str, docs: list[Document]):
        question_prompt = PromptTemplate(
            template=dedent(
                """
                A series of user interviews were conducted to try and answer the question: "{query}".
                Your job is to answer the question by summarizing the responses across all interviews.

                Here is the first interview:

                {text}

                The summary so far is:
                """
            ),
            input_variables=["text", "query"],
        )
        refine_prompt = PromptTemplate(
            template=dedent(
                """
                A series of user interviews were conducted to try and answer the question: "{query}"
                Your job is to answer the question by summarizing the responses across all interviews.

                The summary so far is:
                {existing_answer}

                Here is the next interview:

                {text}

                The new summary so far is:
                """
            ),
            input_variables=["text", "query", "existing_answer"],
        )
        chain = load_summarize_chain(
            self.llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
        )
        return self.cached(
            "summarize_facts",
            chain,
            docs,
            lambda d: {"input_documents": d, "query": query},
        ).strip()
