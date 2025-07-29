# rag_index.py
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, os

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

#  해외 뉴스 JSON → 벡터DB 생성
def create_faiss_index(keyword):
    json_path = f"data/{keyword}.json"
    if not os.path.exists(json_path):
        print(f"❌ {json_path} 없음")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [
        item.get("preview", "").strip()
        for item in data
        if item.get("preview") and len(item["preview"].strip()) > 50
    ]
    if not docs:
        print("❌ 유효한 뉴스 문서 없음")
        return

    embs = embed_model.encode(docs)
    if len(embs.shape) == 1:
        embs = embs.reshape(1, -1)

    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(np.array(embs))

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(idx, f"embeddings/{keyword}_index.faiss")
    print(f"뉴스 벡터DB 저장: embeddings/{keyword}_index.faiss")

# 한국 공시/뉴스 → 직접 docs 리스트 받아 벡터DB 생성
def create_faiss_index_from_docs(docs, save_path):
    if not docs or len(docs) == 0:
        raise ValueError("❌ docs 리스트 비어있음")

    embs = embed_model.encode(docs)
    if len(embs.shape) == 1:
        embs = embs.reshape(1, -1)

    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(np.array(embs))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(idx, save_path)
    return idx