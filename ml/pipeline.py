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
from models import train_all_models, compute_cv_summary
from metrics import compute_metrics, compute_cm_stats
from recommender import recommend
from visualize import generate_charts


def run_pipeline(source, target_column=None, task_type=None, class_weight_mode="off") -> dict:
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
    class_weight_mode = (class_weight_mode or "off").lower()
    if class_weight_mode not in ("off", "on"):
        raise ValueError("class_weight_mode must be 'off' or 'on'")

    class_weighting_applied = False
    if result.task_type == "classification":
        class_weighting_applied = class_weight_mode == "on"

    models = train_all_models(result, use_class_weighting=class_weighting_applied)

    # Filter out any models that failed to train
    models = {k: v for k, v in models.items() if v is not None}

    if not models:
        raise RuntimeError("All models failed to train. Check your dataset.")

    # ── Step 3: Compute metrics ──────────────────────────────────────────────
    metrics_df = compute_metrics(models, result)

    # ── Step 3c: Cross-validation summary for top models ─────────────────────
    cv_summary = compute_cv_summary(result, metrics_df, models, top_n=3, cv_folds=3)


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
        "cv_summary": cv_summary,
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
            "class_imbalance": result.class_imbalance,
            "leakage_warnings": result.leakage_warnings,
            "class_weight_mode": class_weight_mode,
            "class_weighting_applied": class_weighting_applied,
        },
    }


