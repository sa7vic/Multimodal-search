from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
from transformers import CLIPModel, CLIPProcessor

from config import (
    CLIP_MODEL,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    HYBRID_BM25_W,
    HYBRID_CLIP_W,
    QDRANT_PATH,
    VECTOR_SIZE,
)

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _model, _processor
    if _model is None:
        print(f"loading CLIP on {_device}")
        _model = CLIPModel.from_pretrained(CLIP_MODEL).to(_device)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _model.eval()
    return _model, _processor


_qdrant = None


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _qdrant = QdrantClient(path=QDRANT_PATH)
        existing = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            _qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    return _qdrant


def _embed_text_vec(text):
    model, processor = _load_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        raw = model.text_model(**inputs)
        projected = model.text_projection(raw.pooler_output)
    vec = projected[0].cpu().detach().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embed_image_vec(image):
    model, processor = _load_model()
    inputs = processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        raw = model.vision_model(**inputs)
        projected = model.visual_projection(raw.pooler_output)
    vec = projected[0].cpu().detach().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _qdrant_search(vec, limit):
    client = _get_qdrant()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=limit,
        with_payload=True,
    )
    return result.points


_bm25 = None
_bm25_ids = []


def _build_bm25_index():
    global _bm25, _bm25_ids
    client = _get_qdrant()
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        limit=100_000,
    )
    if not points:
        _bm25 = None
        _bm25_ids = []
        return
    _bm25_ids = [p.id for p in points]
    corpus = []
    for p in points:
        name = p.payload.get("filename", "")
        caption = p.payload.get("caption", "")
        tokens = (name + " " + caption).lower().replace("-", " ").replace("_", " ").split()
        corpus.append(tokens)
    _bm25 = BM25Okapi(corpus)


def _get_bm25_scores(query_tokens):
    if _bm25 is None:
        _build_bm25_index()
    if _bm25 is None or not _bm25_ids:
        return {}
    scores = _bm25.get_scores(query_tokens)
    max_s = scores.max() if scores.max() > 0 else 1.0
    return {pid: float(s / max_s) for pid, s in zip(_bm25_ids, scores)}


def _format_results(hits):
    return [
        {
            "id": h.id,
            "score": round(float(h.score), 4),
            "filename": h.payload.get("filename", ""),
            "caption": h.payload.get("caption", ""),
            "path": h.payload.get("path", ""),
        }
        for h in hits
    ]


def search_by_text(query, top_k=DEFAULT_TOP_K):
    vec = _embed_text_vec(query)
    hits = _qdrant_search(vec, top_k)
    return _format_results(hits)


def search_by_image(image, top_k=DEFAULT_TOP_K):
    vec = _embed_image_vec(image)
    hits = _qdrant_search(vec, top_k)
    return _format_results(hits)


def search_hybrid(query, top_k=DEFAULT_TOP_K):
    vec = _embed_text_vec(query)
    candidates = _qdrant_search(vec, top_k * 4)

    query_tokens = query.lower().split()
    bm25_scores = _get_bm25_scores(query_tokens)

    reranked = []
    for hit in candidates:
        clip_s = float(hit.score)
        bm25_s = bm25_scores.get(hit.id, 0.0)
        combined = HYBRID_CLIP_W * clip_s + HYBRID_BM25_W * bm25_s
        reranked.append({
            "id": hit.id,
            "score": round(combined, 4),
            "clip_score": round(clip_s, 4),
            "bm25_score": round(bm25_s, 4),
            "filename": hit.payload.get("filename", ""),
            "caption": hit.payload.get("caption", ""),
            "path": hit.payload.get("path", ""),
        })

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def get_stats():
    client = _get_qdrant()
    info = client.get_collection(COLLECTION_NAME)
    return {
        "total_images": info.points_count,
        "vector_size": VECTOR_SIZE,
        "clip_model": CLIP_MODEL,
        "device": _device,
        "collection": COLLECTION_NAME,
    }


def invalidate_bm25_cache():
    global _bm25, _bm25_ids
    _bm25 = None
    _bm25_ids = []
