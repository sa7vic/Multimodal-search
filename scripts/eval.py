"""
eval.py
Offline evaluation harness.
Measures Recall@5, Recall@10, and MRR for text-only vs hybrid search.

How to use:
1. Edit QUERIES below — add your own text queries and the filenames of
   known-relevant images for each query.
2. Run:  python scripts/eval.py

The more queries you add, the more meaningful the numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from app.search import search_by_text, search_hybrid

QUERIES: list[tuple[str, list[str]]] = [
    ("sunset over the ocean", ["sunset_ocean.jpg", "golden_horizon.jpg"]),
    ("a dog playing in the park", ["dog_park.jpg", "puppy_grass.jpg"]),
    ("mountain landscape with snow", ["snowy_mountain.jpg", "alpine_view.jpg"]),
    ("coffee cup on a wooden table", ["coffee_wood.jpg", "morning_cup.jpg"]),
    ("city skyline at night", ["city_night.jpg", "skyline_lights.jpg"]),
    ("a red sports car", ["red_car.jpg", "sports_car_red.jpg"]),
    ("children laughing", ["kids_laugh.jpg", "children_happy.jpg"]),
    ("forest with sunlight through trees", ["forest_light.jpg", "sunbeam_trees.jpg"]),
    ("fresh vegetables on a market stall", ["veggies_market.jpg", "fresh_produce.jpg"]),
    ("cat sitting on a window sill", ["cat_window.jpg", "kitty_sill.jpg"]),
    ("people running a marathon", ["marathon_run.jpg", "race_crowd.jpg"]),
    ("aerial view of a beach", ["beach_aerial.jpg", "coastline_top.jpg"]),
    ("a woman reading a book", ["woman_reading.jpg", "reading_bench.jpg"]),
    ("old architecture in europe", ["euro_arch.jpg", "cobblestone_city.jpg"]),
    ("plate of spaghetti", ["spaghetti_plate.jpg", "pasta_dish.jpg"]),
    ("autumn leaves on the ground", ["autumn_leaves.jpg", "fall_foliage.jpg"]),
    ("bicycle parked on a street", ["bike_street.jpg", "parked_bicycle.jpg"]),
    ("lightning storm over a city", ["lightning_city.jpg", "storm_night.jpg"]),
    ("underwater coral reef", ["coral_reef.jpg", "underwater_fish.jpg"]),
    ("snowy village at christmas", ["christmas_village.jpg", "snow_houses.jpg"]),
]


def recall_at_k(results: list[dict], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in results[:k] if r["filename"] in relevant)
    return hits / len(relevant)


def reciprocal_rank(results: list[dict], relevant: set[str]) -> float:
    for rank, r in enumerate(results, start=1):
        if r["filename"] in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(search_fn, queries: list[tuple[str, list[str]]]) -> dict:
    r5_scores, r10_scores, rr_scores = [], [], []

    for query, rel_files in queries:
        relevant = set(rel_files)
        try:
            results = search_fn(query, top_k=20)
        except Exception as e:
            print(f"  [warn] search failed for '{query}': {e}")
            results = []

        r5_scores.append(recall_at_k(results, relevant, 5))
        r10_scores.append(recall_at_k(results, relevant, 10))
        rr_scores.append(reciprocal_rank(results, relevant))

    return {
        "Recall@5":  round(np.mean(r5_scores), 4),
        "Recall@10": round(np.mean(r10_scores), 4),
        "MRR":       round(np.mean(rr_scores), 4),
    }


def plot_comparison(text_metrics: dict, hybrid_metrics: dict, out_path: str) -> None:
    labels  = list(text_metrics.keys())
    text_v  = list(text_metrics.values())
    hybrid_v = list(hybrid_metrics.values())

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, text_v,   w, label="Text-only CLIP", color="#7F77DD")
    ax.bar(x + w / 2, hybrid_v, w, label="Hybrid (CLIP + BM25)", color="#1D9E75")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Text-only CLIP vs Hybrid retrieval", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[eval] Chart saved → {out_path}")


if __name__ == "__main__":
    print(f"[eval] Running evaluation on {len(QUERIES)} queries...\n")

    print("[eval] Text-only CLIP...")
    text_metrics = evaluate(search_by_text, QUERIES)

    print("[eval] Hybrid (CLIP + BM25)...")
    hybrid_metrics = evaluate(search_hybrid, QUERIES)

    print("\n── Results ───────────────────────────────")
    print(f"{'Metric':<12}  {'Text-only':>10}  {'Hybrid':>10}  {'Delta':>8}")
    print("─" * 46)
    for m in text_metrics:
        t = text_metrics[m]
        h = hybrid_metrics[m]
        delta = h - t
        sign  = "+" if delta >= 0 else ""
        print(f"{m:<12}  {t:>10.4f}  {h:>10.4f}  {sign}{delta:.4f}")
    print("─" * 46)

    plot_comparison(text_metrics, hybrid_metrics, out_path="data/eval_results.png")
