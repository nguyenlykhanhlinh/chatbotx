import json
from rank_bm25 import BM25Okapi
import numpy as np
from rapidfuzz import fuzz
from typing import List, Dict, Any


class ProductSearch:
    def __init__(self, data_path: str = "data/data.json"):
        self.data_path = data_path
        self.products = []
        self.bm25 = None
        self.load_data()
        self.create_index()

    def load_data(self):
        """Load product data from JSON file"""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.products = json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.products = []

    def create_index(self):
        """Create BM25 index from product data"""
        corpus = []
        for product in self.products:
            # Kết hợp title và content để tìm kiếm
            text = f"{product['title']} {product.get('content', '')}"
            # Tokenize đơn giản bằng cách split space
            tokens = text.lower().split()
            corpus.append(tokens)

        # Tạo BM25 index
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search products using BM25 + Fuzzy matching
        Args:
            query: Search query
            top_k: Number of results to return
        Returns:
            List of products with scores
        """
        # BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())

        # Fuzzy search
        results = []
        for i, product in enumerate(self.products):
            # Tính điểm fuzzy match với title
            fuzzy_score = (
                fuzz.token_sort_ratio(query.lower(), product["title"].lower()) / 100
            )

            # Kết hợp điểm BM25 và Fuzzy
            final_score = (
                0.7 * bm25_scores[i] + 0.3 * fuzzy_score  # BM25: 70%  # Fuzzy: 30%
            )

            if final_score > 0.5:  # Chỉ lấy kết quả có điểm > 50%
                results.append({**product, "score": final_score})

        # Sắp xếp theo điểm và lấy top_k kết quả
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


# Singleton instance
product_search = ProductSearch()
