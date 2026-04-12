"""
test_pipeline.py
================
End-to-end tests for the full ML pipeline.
Tests both regression and classification flows with inline CSVs
and a synthetic noisy dataset.

Run:  python test_pipeline.py
"""

import io
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_pipeline


def test_regression():
    """Test full pipeline on a regression dataset."""
    print("TEST 1: Regression pipeline")
    print("-" * 50)

    csv = """id,age,salary,experience,years_education,department,target_income
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

    output = run_pipeline(io.StringIO(csv), target_column="target_income", task_type="regression")

    # Assert output structure
    assert "preprocess_log" in output, "Missing preprocess_log"
    assert "metrics" in output, "Missing metrics"
    assert "recommendation" in output, "Missing recommendation"
    assert "charts" in output, "Missing charts"
    assert "dataset_info" in output, "Missing dataset_info"

    # Assert metrics
    assert len(output["metrics"]) > 0, "No metrics computed"
    first_metric = output["metrics"][0]
    assert "model_name" in first_metric, "Missing model_name in metrics"
    assert "train_mse" in first_metric, "Missing train_mse (regression)"
    assert "test_r2" in first_metric, "Missing test_r2 (regression)"
    assert "fit_label" in first_metric, "Missing fit_label"

    # Assert recommendation
    rec = output["recommendation"]
    assert "best_model" in rec, "Missing best_model"
    assert "verdict" in rec, "Missing verdict"
    assert "overfit_models" in rec, "Missing overfit_models"
    assert "underfit_models" in rec, "Missing underfit_models"
    assert len(rec["verdict"]) > 50, "Verdict too short"

    # Assert charts
    assert len(output["charts"]) >= 5, f"Expected >= 5 charts, got {len(output['charts'])}"
    for chart_name, b64 in output["charts"].items():
        assert len(b64) > 100, f"Chart '{chart_name}' seems empty"

    # Assert dataset_info
    info = output["dataset_info"]
    assert info["task_type"] == "regression", "Task type should be regression"
    assert len(info["feature_names"]) > 0, "No features"

    print(f"  [OK] Output keys: {list(output.keys())}")
    print(f"  [OK] Models scored: {len(output['metrics'])}")
    print(f"  [OK] Charts: {list(output['charts'].keys())}")
    print(f"  [OK] Best model: {rec['best_model']}")
    print(f"  [OK] Task type: {info['task_type']}")
    print("  PASSED\n")


def test_classification():
    """Test full pipeline on a classification dataset."""
    print("TEST 2: Classification pipeline")
    print("-" * 50)

    csv = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
6.7,3.1,4.7,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
5.7,2.8,4.5,1.3,versicolor
6.4,3.2,4.5,1.5,versicolor
5.2,3.5,1.5,0.2,setosa
7.7,3.8,6.7,2.2,virginica
5.5,2.4,3.8,1.1,versicolor
4.6,3.1,1.5,0.2,setosa
6.9,3.1,5.1,2.3,virginica
5.0,3.4,1.5,0.2,setosa
6.1,2.9,4.7,1.4,versicolor
7.2,3.2,6.0,1.8,virginica
5.4,3.7,1.5,0.2,setosa
6.5,3.0,5.5,1.8,virginica
5.6,2.9,3.6,1.3,versicolor
4.8,3.4,1.6,0.2,setosa
7.4,2.8,6.1,1.9,virginica"""

    output = run_pipeline(io.StringIO(csv), target_column="species")

    # Assert classification-specific metrics
    first_metric = output["metrics"][0]
    assert "train_accuracy" in first_metric, "Missing train_accuracy (classification)"
    assert "test_f1" in first_metric, "Missing test_f1 (classification)"

    # Assert classification charts
    assert "confusion_matrix" in output["charts"], "Missing confusion_matrix chart"

    # Assert dataset_info
    info = output["dataset_info"]
    assert info["task_type"] == "classification", "Task type should be classification"
    assert info["n_classes"] >= 2, "Should have at least 2 classes"

    rec = output["recommendation"]
    print(f"  [OK] Models scored: {len(output['metrics'])}")
    print(f"  [OK] Charts: {list(output['charts'].keys())}")
    print(f"  [OK] Best model: {rec['best_model']}")
    print(f"  [OK] Classes: {info['class_labels']} ({info['n_classes']})")
    print("  PASSED\n")


def test_synthetic_noisy():
    """Test with a synthetic noisy dataset generated via numpy."""
    print("TEST 3: Synthetic noisy regression dataset")
    print("-" * 50)

    np.random.seed(42)
    n = 200
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    noise = np.random.randn(n) * 5
    y = 3 * X1 + 2 * X2 - X3 + noise  # linear with noise

    df = pd.DataFrame({"feat_1": X1, "feat_2": X2, "feat_3": X3, "target": y})

    # Save to a StringIO as CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    output = run_pipeline(csv_buf, target_column="target", task_type="regression")

    # With 200 noisy samples, we expect:
    # - Some models to overfit (DT unlimited, KNN_k1)
    # - Some to underfit (DT_depth1)
    # - Some to be good (Linear, Ridge — since true relationship is linear)
    rec = output["recommendation"]
    metrics = output["metrics"]

    print(f"  [OK] Models scored: {len(metrics)}")
    print(f"  [OK] Best model: {rec['best_model']}")
    print(f"  [OK] Overfit: {rec['overfit_models']}")
    print(f"  [OK] Underfit: {rec['underfit_models']}")

    # The best model should have a decent test R2 (true relationship is linear)
    best_metrics = [m for m in metrics if m["model_name"] == rec["best_model"]][0]
    print(f"  [OK] Best test R2: {best_metrics['test_r2']:.4f}")
    assert best_metrics["test_r2"] > 0.3, "Best model R2 too low on synthetic data"

    print("  PASSED\n")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FULL PIPELINE TESTS")
    print("=" * 60 + "\n")

    test_regression()
    test_classification()
    test_synthetic_noisy()

    print("=" * 60)
    print("ALL 3 TESTS PASSED")
    print("=" * 60)
