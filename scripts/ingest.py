import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import (
    CLIP_MODEL,
    COLLECTION_NAME,
    DATA_DIR,
    INGEST_BATCH_SIZE,
    QDRANT_PATH,
    SUPPORTED_EXTS,
    VECTOR_SIZE,
)


def collect_image_paths(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                paths.append(Path(dirpath) / fn)
    return paths


def embed_batch(paths, model, processor, device):
    images = []
    valid_paths = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_paths.append(p)
        except Exception:
            print(f"  skipping {p}")

    if not images:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32), []

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        raw = model.vision_model(**inputs)
        pooled = raw.pooler_output
        projected = model.visual_projection(pooled)

    vecs = projected.cpu().detach().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vecs / norms, valid_paths


def ingest(data_dir, batch_size, reset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing = []

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    all_paths = collect_image_paths(data_dir)
    if not all_paths:
        print(f"no images found in {data_dir}")
        return

    print(f"found {len(all_paths)} images")

    point_id = 0
    for i in tqdm(range(0, len(all_paths), batch_size), unit="batch"):
        batch_paths = all_paths[i: i + batch_size]
        vecs, valid = embed_batch(batch_paths, model, processor, device)

        points = []
        for vec, path in zip(vecs, valid):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "filename": path.name,
                        "path": str(path.resolve()),
                        "caption": path.stem.replace("_", " ").replace("-", " "),
                    },
                )
            )
            point_id += 1

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"indexed {point_id} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=INGEST_BATCH_SIZE)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    ingest(args.data_dir, args.batch_size, args.reset)
