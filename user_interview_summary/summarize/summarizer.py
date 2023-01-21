from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from user_interview_summary.shared.chain import Chain


class Summarizer(Chain):
    def summarize_class(self, docs: list[Document]):
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(docs)  # type: ignore

    def summarize_file(self, docs: list[Document]):
        chain = load_summarize_chain(self.llm, chain_type="refine")
        return chain.run(docs)  # type: ignore

    def summarize_doc(self, doc: Document):
        chain = load_summarize_chain(self.llm, chain_type="stuff")
        return chain.run(doc)  # type: ignore
