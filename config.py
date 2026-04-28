import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "images")
QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_store")

COLLECTION_NAME = "images"
VECTOR_SIZE     = 512

CLIP_MODEL = "openai/clip-vit-base-patch32"

DEFAULT_TOP_K  = 12
HYBRID_CLIP_W  = 0.75
HYBRID_BM25_W  = 0.25

INGEST_BATCH_SIZE = 64
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FLASK_HOST       = "0.0.0.0"
FLASK_PORT       = 5000
FLASK_DEBUG      = True
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
