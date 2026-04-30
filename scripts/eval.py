"""
eval.py
Offline evaluation harness.
Measures Recall@5, Recall@10, and MRR for text-only vs hybrid search.

How to use:
1. Edit QUERIES below — add your own text queries and the filenames of
   known-relevant images for each query.
2. Run:  python scripts/eval.py
3. A bar chart is saved to data/eval_results.png

The more queries you add, the more meaningful the numbers.
"""