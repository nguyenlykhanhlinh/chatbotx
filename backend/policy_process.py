import os
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from utils.retriever import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PolicySearch(BaseRetriever):
    def __init__(self):
        super().__init__()
        self.setup_components()
        self.documents = self._load_and_process_documents()
        self.setup_retrievers()

    def setup_components(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _load_and_process_documents(self):
        with open("backend/luxy.txt", "r", encoding="utf-8") as f:
            content = f.read()

        doc = Document(page_content=content)
        chunks = self.text_splitter.create_documents([doc.page_content])

        return [chunk for chunk in chunks if len(chunk.page_content.split()) > 5]

    def setup_retrievers(self):
        vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.hybrid_retriever = self.setup_hybrid_retriever(
            self.documents, vector_store
        )

    def get_policy_response(self, query: str) -> str:
        POLICY_PROMPT = """Bạn là trợ lý chăm sóc khách hàng của Siêu Thị Luxy.
        Sử dụng thông tin trong context để trả lời câu hỏi về chính sách và dịch vụ.
        Nếu không tìm thấy thông tin trong context, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin về vấn đề này."
        
        Context: {context}
        Câu hỏi: {question}"""

        return self.get_response(query, POLICY_PROMPT, self.hybrid_retriever)


# Singleton instance
policy_search = PolicySearch()
