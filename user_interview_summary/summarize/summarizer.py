from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from user_interview_summary.shared.chain import Chain, LChain


class Summarizer(Chain):
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
