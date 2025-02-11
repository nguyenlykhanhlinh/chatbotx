from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import FAISS, Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os


class BaseRetriever:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.3)

    def setup_hybrid_retriever(self, documents, vector_store):
        bm25_retriever = BM25Retriever.from_documents(documents, k=5)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
        )

    def get_response(self, query: str, prompt_template: str, retriever) -> str:
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        return chain.run(context=context, question=query)
