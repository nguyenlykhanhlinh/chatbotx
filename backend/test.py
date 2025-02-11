from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi
import numpy as np
import json

# Giả lập knowledge base
knowledge_base = [
    {
        "question": "Làm thế nào để làm bánh mì?",
        "answer": "Để làm bánh mì bạn cần bột mì, men, muối và nước. Trộn các nguyên liệu và ủ bột.",
        "keywords": ["bánh mì", "bột mì", "men", "muối", "nước", "ủ bột"]
    },
    {
        "question": "Cách làm bánh ngọt?",
        "answer": "Bánh ngọt cần bột mì, trứng, đường và bơ. Trộn đều và nướng ở 180 độ.",
        "keywords": ["bánh ngọt", "bột mì", "trứng", "đường", "bơ", "nướng"]
    },
    {
        "question": "Công thức làm bánh bông lan?",
        "answer": "Bánh bông lan cần 4 quả trứng, 100g đường, 100g bột mì và bơ.",
        "keywords": ["bánh bông lan", "trứng", "đường", "bột mì", "bơ"]
    },
    {
        "question": "Hướng dẫn làm bánh flan?",
        "answer": "Bánh flan cần trứng, sữa, đường và caramel. Hấp cách thủy 20 phút.",
        "keywords": ["bánh flan", "trứng", "sữa", "đường", "caramel", "hấp"]
    }
]

class SimpleRetriever:
    def __init__(self, documents, bm25_weight=0.7, fuzzy_weight=0.3):
        """
        Khởi tạo retriever với 2 phương pháp tìm kiếm:
        - BM25: Tìm kiếm dựa trên tần suất từ và độ dài văn bản
        - Fuzzy: Xử lý lỗi chính tả
        """
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.fuzzy_weight = fuzzy_weight
        
        # Chuẩn bị corpus cho BM25
        self.corpus = [
            ' '.join([doc['question']] + doc['keywords'])
            for doc in documents
        ]
        self.bm25 = BM25Okapi(
            [text.lower().split() for text in self.corpus]
        )

    def search(self, query, top_k=3, threshold=0.3):
        """
        Tìm kiếm kết hợp BM25 và fuzzy matching
        """
        # 1. Tìm với BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = np.array(bm25_scores) / max(bm25_scores) if max(bm25_scores) > 0 else bm25_scores
        
        # 2. Tìm với fuzzy matching
        fuzzy_scores = []
        for doc in self.documents:
            # Kết hợp điểm của question và keywords
            question_score = fuzz.token_sort_ratio(query, doc['question']) / 100
            keyword_scores = [
                fuzz.token_sort_ratio(query, keyword) / 100
                for keyword in doc['keywords']
            ]
            max_keyword_score = max(keyword_scores)
            fuzzy_scores.append((question_score + max_keyword_score) / 2)
        fuzzy_scores = np.array(fuzzy_scores)
        
        # Kết hợp điểm từ 2 phương pháp
        combined_scores = (
            self.bm25_weight * bm25_scores +
            self.fuzzy_weight * fuzzy_scores
        )
        
        # Lọc và sắp xếp kết quả
        results = []
        for i, score in enumerate(combined_scores):
            if score > threshold:
                results.append({
                    'question': self.documents[i]['question'],
                    'answer': self.documents[i]['answer'],
                    'score': score,
                    'bm25_score': bm25_scores[i],
                    'fuzzy_score': fuzzy_scores[i]
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

def test_simple_search():
    print("\n🔍 Test Simple Search (BM25 + Fuzzy):")
    
    # Khởi tạo retriever
    retriever = SimpleRetriever(knowledge_base)
    
    # Test case 1: Query chính xác
    query = "Làm thế nào để làm bánh mì?"
    print(f"\nTest 1 - Query chính xác:")
    print(f"Query: '{query}'")
    results = retriever.search(query)
    for i, result in enumerate(results, 1):
        print(f"\nKết quả #{i}:")
        print(f"Câu hỏi: {result['question']}")
        print(f"Câu trả lời: {result['answer']}")
        print(f"Điểm tổng hợp: {result['score']:.2f}")
        print(f"Điểm BM25: {result['bm25_score']:.2f}")
        print(f"Điểm Fuzzy: {result['fuzzy_score']:.2f}")

    # Test case 2: Query không chuẩn
    query = "cach nau banh mi ngon"
    print(f"\nTest 2 - Query không chuẩn:")
    print(f"Query: '{query}'")
    results = retriever.search(query)
    for i, result in enumerate(results, 1):
        print(f"\nKết quả #{i}:")
        print(f"Câu hỏi: {result['question']}")
        print(f"Câu trả lời: {result['answer']}")
        print(f"Điểm tổng hợp: {result['score']:.2f}")
        print(f"Điểm BM25: {result['bm25_score']:.2f}")
        print(f"Điểm Fuzzy: {result['fuzzy_score']:.2f}")

    # Test case 3: Query ngữ nghĩa
    query = "hướng dẫn làm món bánh ngọt từ bột mì và trứng"
    print(f"\nTest 3 - Query ngữ nghĩa:")
    print(f"Query: '{query}'")
    results = retriever.search(query)
    for i, result in enumerate(results, 1):
        print(f"\nKết quả #{i}:")
        print(f"Câu hỏi: {result['question']}")
        print(f"Câu trả lời: {result['answer']}")
        print(f"Điểm tổng hợp: {result['score']:.2f}")
        print(f"Điểm BM25: {result['bm25_score']:.2f}")
        print(f"Điểm Fuzzy: {result['fuzzy_score']:.2f}")

    print("\n📝 Giải thích:")
    print("1. BM25 (70%): Tốt cho tìm kiếm dựa trên tần suất từ và độ dài văn bản")
    print("2. Fuzzy (30%): Tốt cho xử lý lỗi chính tả và dấu")
    print("3. Có thể điều chỉnh trọng số tùy use case")

if __name__ == "__main__":
    try:
        print("🚀 Bắt đầu test simple search...")
        test_simple_search()
        print("\n✅ Test hoàn thành!")
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
    finally:
        print("\n🏁 Kết thúc test")
