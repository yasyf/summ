from langchain.chains.summarize import load_summarize_chain

from user_interview_summary.shared.chain import Chain


class Summarizer(Chain):
    def summarize_class(self, docs):
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(docs)

    def summarize_file(self, docs):
        chain = load_summarize_chain(self.llm, chain_type="refine")
        return chain.run(docs)
