"""
pipeline.py
============
Single orchestrator that chains the entire ML pipeline:
    preprocess -> train -> metrics -> recommend -> visualize

This is the ONE function the Flask backend calls. No ML logic lives here —
it just wires the four modules together and returns a complete result dict.

Usage:
    from pipeline import run_pipeline

    result = run_pipeline("dataset.csv", target_column="price")
    # result = {"preprocess_log": [...], "metrics": [...], "recommendation": {...}, ...}
"""

import sys
import os

# Ensure ml/ directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessor import preprocess
from models import train_all_models
from metrics import compute_metrics, compute_baseline, compute_cm_stats
from recommender import recommend
from visualize import generate_charts


def run_pipeline(source, target_column=None, task_type=None) -> dict:
    """
    Run the full ML pipeline end-to-end.

    Parameters
    ----------
    source : str or file-like
        Path to CSV file, or a file-like object (e.g. uploaded file).
    target_column : str, optional
        Name of the target column. If None, last column is used.
    task_type : str, optional
        Force "regression" or "classification". None = auto-detect.

    Returns
    -------
    dict with keys:
        preprocess_log  : list[str]  — step-by-step log from preprocessor
        metrics         : list[dict] — one dict per model (DataFrame rows)
        recommendation  : dict       — best_model, verdict, overfit/underfit lists
        charts          : dict       — {"chart_name": "base64_png_string", ...}
        dataset_info    : dict       — shapes, feature names, target, task type
    """
    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    result = preprocess(source, target_column=target_column, task_type=task_type)

    # ── Step 2: Train all models ─────────────────────────────────────────────
    models = train_all_models(result)

    # Filter out any models that failed to train
    models = {k: v for k, v in models.items() if v is not None}

    if not models:
        raise RuntimeError("All models failed to train. Check your dataset.")

    # ── Step 3: Compute metrics ──────────────────────────────────────────────
    metrics_df = compute_metrics(models, result)

    # ── Step 3b: Compute naive baseline ──────────────────────────────────────
    baseline = compute_baseline(result)

    # ── Step 4: Recommend best model ─────────────────────────────────────────
    recommendation = recommend(metrics_df, result=result)

    # ── Step 4b: Confusion matrix insights (classification only) ─────────────
    best_model_obj = models.get(recommendation["best_model"])
    cm_stats = compute_cm_stats(best_model_obj, result)

    # ── Step 5: Generate charts ──────────────────────────────────────────────
    charts = generate_charts(result, models, metrics_df, recommendation["best_model"])

    # ── Step 6: Package output ───────────────────────────────────────────────
    return {
        "preprocess_log": result.log,
        "metrics": metrics_df.to_dict(orient="records"),
        "baseline": baseline,
        "cm_stats": cm_stats,
        "recommendation": recommendation,
        "charts": charts,
        "dataset_info": {
            "original_shape": list(result.original_shape),
            "final_shape": list(result.final_shape),
            "feature_names": result.feature_names,
            "target_column": result.target_column,
            "task_type": result.task_type,
            "n_classes": result.n_classes,
            "class_labels": result.class_labels,
            "class_balance": result.class_balance,
        },
    }


# ─────────────────────────────────────────────
# QUICK TEST  (run: python pipeline.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import io

    reg_csv = """id,age,salary,experience,years_education,department,target_income
1,25,50000,2,16,Engineering,55000
2,32,75000,8,18,Marketing,80000
3,28,60000,5,16,Engineering,65000
4,45,95000,20,20,Management,100000
5,38,85000,14,18,Engineering,90000
6,29,62000,6,16,Marketing,67000
7,52,110000,28,22,Management,115000
8,35,78000,11,18,Engineering,83000
9,26,53000,3,16,Marketing,58000
10,41,92000,17,20,Engineering,97000
11,33,72000,9,18,Management,77000
12,27,56000,4,16,Engineering,61000
13,48,105000,24,20,Marketing,110000
14,36,81000,12,18,Engineering,86000
15,30,67000,7,16,Management,72000"""

    print("=" * 60)
    print("PIPELINE: FULL END-TO-END TEST")
    print("=" * 60)

    output = run_pipeline(io.StringIO(reg_csv), target_column="target_income", task_type="regression")

    print(f"\n[OK] Keys: {list(output.keys())}")
    print(f"[OK] Preprocess log: {len(output['preprocess_log'])} entries")
    print(f"[OK] Models scored: {len(output['metrics'])}")
    print(f"[OK] Charts generated: {len(output['charts'])} -> {list(output['charts'].keys())}")
    print(f"[OK] Best model: {output['recommendation']['best_model']}")
    print(f"[OK] Task type: {output['dataset_info']['task_type']}")
    print(f"[OK] Features: {output['dataset_info']['feature_names']}")

    print("\n" + "=" * 60)
    print("PIPELINE TEST PASSED")
    print("=" * 60)
