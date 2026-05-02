# Multimodal Search (CLIP + Qdrant + BM25)

A multimodal image retrieval system that supports:

- **Text → Image** semantic search (natural-language queries)
- **Image → Image** similarity search (query by example)
- **Hybrid retrieval** that combines **CLIP semantic similarity** with **BM25 lexical scoring** over metadata for more robust results

The project includes an ingestion pipeline for building a local vector index, a Flask API, and a clean web UI for interactive search.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [System Overview](#system-overview)
- [Retrieval Methods](#retrieval-methods)
  - [Text Search](#text-search)
  - [Image Search](#image-search)
  - [Hybrid Search (CLIP + BM25)](#hybrid-search-clip--bm25)
- [API](#api)
- [Local Setup](#local-setup)
- [Indexing / Ingestion](#indexing--ingestion)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Implementation Notes](#implementation-notes)

---

## Key Capabilities

- **CLIP embeddings** via Hugging Face Transformers (`openai/clip-vit-base-patch32`)
- **Vector search** with **Qdrant** using **cosine similarity** over 512-d normalized vectors
- **Hybrid reranking** that fuses:
  - semantic similarity from CLIP (dense vectors)
  - lexical relevance from BM25 (sparse term scoring over `filename + caption`)
- **Local-first storage**: Qdrant runs in embedded mode (persisted to disk)
- **Web UI** with 3 modes: Text / Hybrid / Image, plus result scores and indexed-image counter
- **Offline evaluation harness** comparing text-only vs hybrid with:
  - **Recall@5**
  - **Recall@10**
  - **MRR**
  - and a saved comparison chart (`data/eval_results.png`)

---

## System Overview

### Data flow

1. **Ingest images** from a directory (`data/images/` by default)
2. Compute **CLIP image embeddings**
3. Store vectors + payload in **Qdrant**
4. Query time:
   - embed text or image
   - retrieve nearest neighbors from Qdrant
   - optionally rerank with BM25 for hybrid mode
5. Serve results via Flask + UI

### Stored payload per image

During ingestion, each indexed point contains:
- `filename`: original file name
- `path`: absolute path to the image file
- `caption`: simple caption derived from the filename (underscores/dashes normalized)

---

## Retrieval Methods

### Text Search

**Goal:** retrieve images that best match a text description.

- Text query is embedded using CLIP’s **text encoder**
- Vector is normalized and searched in Qdrant using cosine similarity
- Returns top-*k* results with metadata (`filename`, `caption`, `path`) and similarity score

### Image Search

**Goal:** retrieve visually similar images using a query image.

- Uploaded image is decoded with Pillow, converted to RGB
- Embedded using CLIP’s **vision encoder**
- Vector is normalized and searched in Qdrant
- Returns top-*k* similar images

### Hybrid Search (CLIP + BM25)

Hybrid retrieval improves behavior on queries where *semantic similarity alone* can be overly broad by adding a lexical signal over metadata.

**Process:**
1. Use **CLIP text embedding** to retrieve a broader candidate set from Qdrant (top `k * 4`)
2. Compute a BM25 score using tokens from:
   - `filename`
   - `caption`
3. Combine scores:
   - `combined = (HYBRID_CLIP_W * clip_score) + (HYBRID_BM25_W * bm25_score)`
4. Sort by combined score and return top-*k*

Weights are configurable in `config.py`:
- `HYBRID_CLIP_W = 0.75`
- `HYBRID_BM25_W = 0.25`

---

## API

### UI
- `GET /`  
  Serves the web UI (`templates/index.html`).

### Search endpoints
- `POST /search/text`  
  **Body (JSON):**
  ```json
  { "query": "sunset over the ocean", "top_k": 20 }
  ```

- `POST /search/hybrid`  
  **Body (JSON):**
  ```json
  { "query": "red sports car", "top_k": 20 }
  ```

- `POST /search/image`  
  **Form-data:**
  - `file`: image file
  - `top_k`: integer (optional)

### Utility endpoints
- `GET /stats`  
  Returns collection statistics (total images, vector size, device, model name).

- `GET /image/<point_id>`  
  Serves an indexed image by id. Includes path resolution logic to locate the image even if absolute paths differ across machines.

---

## Local Setup

### Requirements
- Python 3.10+ recommended
- PyTorch (CPU or CUDA; the app automatically uses GPU if available)

### Install

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Indexing / Ingestion

Place images under:

```text
data/images/
```

Build the index:

```bash
python scripts/ingest.py --data_dir data/images --reset
```

Notes:
- `--reset` deletes and recreates the Qdrant collection before indexing.
- Vectors persist locally at `data/qdrant_store/`.
- Supported extensions are defined in `config.py`:
  `.jpg, .jpeg, .png, .webp, .bmp`

---

## Run the Web App

This repository uses a Flask app factory (`app/create_app`). A simple way to run:

```bash
python -c "from app import create_app; create_app().run(host='0.0.0.0', port=5000, debug=True)"
```

Then open:

- http://localhost:5000

---

## Evaluation

Run the offline evaluation harness:

```bash
python scripts/eval.py
```

What it does:
- Executes a set of predefined text queries (edit `QUERIES` in `scripts/eval.py`)
- Compares:
  - **Text-only CLIP** retrieval
  - **Hybrid (CLIP + BM25)** retrieval
- Computes:
  - `Recall@5`
  - `Recall@10`
  - `MRR`
- Generates a bar chart saved to:
  - `data/eval_results.png`

To make results meaningful, update `QUERIES` with:
- your own natural-language queries
- the filenames of known-relevant images for each query

---

## Configuration

All key knobs are centralized in `config.py`:

- **Data & storage**
  - `DATA_DIR`: image directory (default `data/images`)
  - `QDRANT_PATH`: local Qdrant persistence directory

- **Model**
  - `CLIP_MODEL = "openai/clip-vit-base-patch32"`
  - `VECTOR_SIZE = 512`

- **Search**
  - `DEFAULT_TOP_K`
  - `HYBRID_CLIP_W`, `HYBRID_BM25_W`

- **Ingestion**
  - `INGEST_BATCH_SIZE`
  - `SUPPORTED_EXTS`

- **Server**
  - `FLASK_HOST`, `FLASK_PORT`, `FLASK_DEBUG`
  - `MAX_UPLOAD_BYTES` (10MB)

---

## Project Structure

```text
.
├── app/
│   ├── __init__.py        # Flask app factory + configuration
│   ├── routes.py          # UI + API routes (text/image/hybrid + stats + image serving)
│   ├── search.py          # CLIP embedding, Qdrant client, BM25 hybrid reranking
│   └── utils.py           # image decoding + file type validation
├── scripts/
│   ├── ingest.py          # embedding + indexing pipeline into Qdrant
│   └── eval.py            # evaluation metrics (Recall@K, MRR) + plotting
├── templates/
│   └── index.html         # UI + client-side rendering logic
├── static/
│   └── style.css          # UI styling
├── config.py              # all tunables in one place
└── requirements.txt
```

---

## Implementation Notes

- **Normalized embeddings:** both text and image vectors are L2-normalized before indexing/search. With normalization, cosine similarity behaves consistently and is stable for ranking.
- **Device selection:** embedding runs on `cuda` if available, otherwise CPU.
- **BM25 caching:** BM25 index is built from existing Qdrant payloads and cached in memory. (Useful for fast hybrid reranking once the collection is loaded.)
- **Robust image serving:** the `/image/<id>` route attempts multiple strategies to resolve the stored file path (exact path, flat lookup in `DATA_DIR`, and recursive scan), improving portability across environments.

---
