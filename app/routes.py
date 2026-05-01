import os
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from app.search import get_stats, search_by_image, search_by_text, search_hybrid, _get_qdrant
from app.utils import is_supported_image, load_image_from_bytes
from config import DEFAULT_TOP_K, COLLECTION_NAME, DATA_DIR

bp = Blueprint("main", __name__)


def _resolve(stored_path):
    """
    Try multiple strategies to find the image file:
    1. Exact stored path (works if running on the same machine it was indexed on)
    2. Filename only inside DATA_DIR flat
    3. Recursive walk of DATA_DIR
    """
    if stored_path:
        p = Path(stored_path)
        if p.is_file():
            return str(p)
        filename = p.name
        candidate = Path(DATA_DIR) / filename
        if candidate.is_file():
            return str(candidate)
        for root, _, files in os.walk(DATA_DIR):
            if filename in files:
                return os.path.join(root, filename)
    return None


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/search/text")
def text_search():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    top_k = min(int(body.get("top_k", DEFAULT_TOP_K)), 50)
    results = search_by_text(query, top_k=top_k)
    return jsonify({"results": results, "mode": "text", "query": query})


@bp.post("/search/image")
def image_search():
    if "file" not in request.files:
        return jsonify({"error": "file field is required"}), 400
    f = request.files["file"]
    if not f.filename or not is_supported_image(f.filename):
        return jsonify({"error": "unsupported file type"}), 415
    top_k = min(int(request.form.get("top_k", DEFAULT_TOP_K)), 50)
    try:
        img = load_image_from_bytes(f.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    results = search_by_image(img, top_k=top_k)
    return jsonify({"results": results, "mode": "image"})


@bp.post("/search/hybrid")
def hybrid_search():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    top_k = min(int(body.get("top_k", DEFAULT_TOP_K)), 50)
    results = search_hybrid(query, top_k=top_k)
    return jsonify({"results": results, "mode": "hybrid", "query": query})


@bp.get("/image/<int:point_id>")
def serve_image(point_id):
    client = _get_qdrant()
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
    )
    if not points:
        abort(404)
    stored_path = points[0].payload.get("path", "")
    resolved = _resolve(stored_path)
    if not resolved:
        abort(404)
    return send_file(resolved)


@bp.get("/debug/image/<int:point_id>")
def debug_image(point_id):
    client = _get_qdrant()
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
    )
    if not points:
        return jsonify({"error": "point not found"}), 404
    stored = points[0].payload.get("path", "")
    resolved = _resolve(stored)
    return jsonify({
        "point_id": point_id,
        "stored_path": stored,
        "resolved_path": resolved,
        "data_dir": DATA_DIR,
        "exists": os.path.isfile(resolved) if resolved else False,
    })


@bp.get("/stats")
def stats():
    return jsonify(get_stats())


@bp.app_errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

@bp.app_errorhandler(413)
def too_large(e):
    return jsonify({"error": "file too large (max 10 MB)"}), 413

@bp.app_errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal server error"}), 500
