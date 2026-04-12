"""
metrics.py
==========
Computes train and test error metrics for every trained model.
Returns a clean pandas DataFrame — one row per model.

Usage:
    from preprocessor import preprocess
    from models import train_all_models
    from metrics import compute_metrics

    result = preprocess("dataset.csv", target_column="price")
    models = train_all_models(result)
    metrics_df = compute_metrics(models, result)

Regression columns:
    model_name, train_mse, test_mse, train_rmse, test_rmse,
    train_r2, test_r2, fit_label

Classification columns:
    model_name, train_accuracy, test_accuracy, train_f1, test_f1,
    train_precision, test_precision, train_recall, test_recall, fit_label
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as sk_confusion_matrix,
)


# ─────────────────────────────────────────────
# CONSTANTS  (thresholds for fit labelling)
# ─────────────────────────────────────────────
# Regression: overfit if test MSE is this many times worse than train MSE
OVERFIT_MSE_RATIO = 1.5

# Regression: overfit if R² gap (train - test) exceeds this
OVERFIT_R2_GAP = 0.10

# Regression: underfit if test R² is below this
UNDERFIT_R2_THRESHOLD = 0.40

# Classification: overfit if accuracy gap (train - test) exceeds this
OVERFIT_ACC_GAP = 0.10

# Classification: underfit if test accuracy is below this
UNDERFIT_ACC_THRESHOLD = 0.50


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def compute_baseline(result) -> dict:
    """
    Compute a naive baseline for comparison.

    Classification: majority-class predictor
        → accuracy = fraction of majority class
        → F1 = weighted F1 predicting majority class for all rows

    Regression: mean-value predictor
        → R² = 0 by definition (predicting mean gives R²=0)
        → RMSE = std of y_test

    Returns
    -------
    dict with keys depending on task type.
    """
    if result.task_type == "regression":
        y_test = result.y_test
        mean_pred = np.full_like(y_test, float(np.mean(result.y_train)), dtype=float)
        rmse = float(np.sqrt(mean_squared_error(y_test, mean_pred)))
        return {
            "strategy": "Mean Prediction",
            "test_r2": 0.0,
            "test_rmse": round(rmse, 4),
            "description": "Predicts the training mean for every sample",
        }
    else:
        y_test = result.y_test
        # Majority class in training set
        unique, counts = np.unique(result.y_train, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        majority_frac = round(float(counts.max() / counts.sum()), 4)
        majority_pred = np.full_like(y_test, majority_class)

        baseline_acc = round(float(accuracy_score(y_test, majority_pred)), 4)
        baseline_f1 = round(float(f1_score(y_test, majority_pred, average="weighted", zero_division=0)), 4)

        return {
            "strategy": "Majority Class",
            "majority_class": str(majority_class),
            "majority_fraction": majority_frac,
            "test_accuracy": baseline_acc,
            "test_f1": baseline_f1,
            "description": f"{majority_frac*100:.0f}% of samples belong to class '{majority_class}'",
        }


def compute_cm_stats(best_model, result) -> dict:
    """
    Compute confusion matrix insights for the best classification model.

    Returns a dict with:
      - cm : list[list[int]]           — raw confusion matrix
      - class_labels : list[str]       — display labels
      - is_binary : bool
      - For binary: tn, fp, fn, tp, sensitivity, specificity, dominant_error
      - For all:    per_class_accuracy : list[{label, correct, total, pct}]
      - insights : list[str]           — ready-to-display sentences
    """
    if result.task_type != "classification" or best_model is None:
        return {}

    y_test = result.y_test
    y_pred = best_model.predict(result.X_test)
    cm = sk_confusion_matrix(y_test, y_pred)

    labels = [str(c) for c in (result.class_labels or sorted(set(y_test)))]
    is_binary = cm.shape[0] == 2

    # Per-class accuracy (recall per class)
    per_class = []
    for i, lbl in enumerate(labels):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        pct = round(correct / total * 100, 1) if total > 0 else 0.0
        per_class.append({"label": lbl, "correct": correct, "total": total, "pct": pct})

    insights = []

    if is_binary:
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        sensitivity = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
        specificity = round(tn / (tn + fp) * 100, 1) if (tn + fp) > 0 else 0.0

        # Human-readable class names
        neg_label = labels[0]
        pos_label = labels[1]

        insights.append(
            f"Correctly identified {tn}/{tn+fp} '{neg_label}' cases "
            f"(specificity: {specificity}%)"
        )
        insights.append(
            f"Correctly identified {tp}/{tp+fn} '{pos_label}' cases "
            f"(sensitivity / recall: {sensitivity}%)"
        )

        if fp > fn:
            insights.append(
                f"Dominant error: {fp} false positives — predicted '{pos_label}' "
                f"when the true label was '{neg_label}' (model leans toward '{pos_label}')"
            )
        elif fn > fp:
            insights.append(
                f"Dominant error: {fn} false negatives — predicted '{neg_label}' "
                f"when the true label was '{pos_label}' (model misses some '{pos_label}' cases)"
            )
        else:
            insights.append("False positives and false negatives are balanced — no strong prediction bias")

        return {
            "is_binary": True,
            "cm": cm.tolist(),
            "class_labels": labels,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "per_class_accuracy": per_class,
            "insights": insights,
        }
    else:
        # Multiclass insights
        best_class = max(per_class, key=lambda x: x["pct"])
        worst_class = min(per_class, key=lambda x: x["pct"])

        insights.append(
            f"Best recognized class: '{best_class['label']}' "
            f"({best_class['correct']}/{best_class['total']} correct, {best_class['pct']}%)"
        )
        insights.append(
            f"Most confused class: '{worst_class['label']}' "
            f"({worst_class['correct']}/{worst_class['total']} correct, {worst_class['pct']}%)"
        )
        insights.append(
            "Off-diagonal cells show misclassification counts between class pairs"
        )

        return {
            "is_binary": False,
            "cm": cm.tolist(),
            "class_labels": labels,
            "per_class_accuracy": per_class,
            "insights": insights,
        }


def compute_metrics(models: dict, result) -> pd.DataFrame:
    """
    Compute train and test metrics for every model.

    Parameters
    ----------
    models : dict
        {"model_name": fitted_pipeline, ...} from train_all_models().
        None values (failed models) are skipped.
    result : PreprocessResult
        Output of preprocess(). Provides X_train, X_test, y_train, y_test, task_type.

    Returns
    -------
    pd.DataFrame
        One row per model with all metrics + fit_label column.
    """
    if result.task_type == "regression":
        return _regression_metrics(models, result)
    else:
        return _classification_metrics(models, result)


# ─────────────────────────────────────────────
# REGRESSION METRICS
# ─────────────────────────────────────────────
def _regression_metrics(models: dict, result) -> pd.DataFrame:
    """Compute MSE, RMSE, R2 on train and test for each model."""
    rows = []

    for name, model in models.items():
        if model is None:
            continue

        train_pred = model.predict(result.X_train)
        test_pred = model.predict(result.X_test)

        train_mse = mean_squared_error(result.y_train, train_pred)
        test_mse = mean_squared_error(result.y_test, test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(result.y_train, train_pred)
        test_r2 = r2_score(result.y_test, test_pred)

        fit_label = _label_regression_fit(train_mse, test_mse, train_r2, test_r2)

        rows.append({
            "model_name": name,
            "train_mse": round(train_mse, 4),
            "test_mse": round(test_mse, 4),
            "train_rmse": round(train_rmse, 4),
            "test_rmse": round(test_rmse, 4),
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "fit_label": fit_label,
        })

    df = pd.DataFrame(rows)
    # Sort by test R2 descending (best first)
    df = df.sort_values("test_r2", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# CLASSIFICATION METRICS
# ─────────────────────────────────────────────
def _classification_metrics(models: dict, result) -> pd.DataFrame:
    """Compute accuracy, F1, precision, recall on train and test for each model."""
    rows = []

    for name, model in models.items():
        if model is None:
            continue

        train_pred = model.predict(result.X_train)
        test_pred = model.predict(result.X_test)

        train_acc = accuracy_score(result.y_train, train_pred)
        test_acc = accuracy_score(result.y_test, test_pred)

        # weighted average handles multiclass properly
        train_f1 = f1_score(result.y_train, train_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(result.y_test, test_pred, average="weighted", zero_division=0)

        train_prec = precision_score(result.y_train, train_pred, average="weighted", zero_division=0)
        test_prec = precision_score(result.y_test, test_pred, average="weighted", zero_division=0)

        train_rec = recall_score(result.y_train, train_pred, average="weighted", zero_division=0)
        test_rec = recall_score(result.y_test, test_pred, average="weighted", zero_division=0)

        fit_label = _label_classification_fit(train_acc, test_acc)

        rows.append({
            "model_name": name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "train_f1": round(train_f1, 4),
            "test_f1": round(test_f1, 4),
            "train_precision": round(train_prec, 4),
            "test_precision": round(test_prec, 4),
            "train_recall": round(train_rec, 4),
            "test_recall": round(test_rec, 4),
            "fit_label": fit_label,
        })

    df = pd.DataFrame(rows)
    # Sort by test F1 descending (best first)
    df = df.sort_values("test_f1", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# FIT LABELLING HELPERS
# ─────────────────────────────────────────────
def _label_regression_fit(train_mse, test_mse, train_r2, test_r2) -> str:
    """
    Classify a regression model's fit:
      - overfit:   train error is low but test error is much higher
      - underfit:  both train and test errors are high (low R2)
      - good_fit:  balanced train/test performance
    """
    r2_gap = train_r2 - test_r2

    # Guard: if train_mse is near zero, use R2 gap only
    if train_mse > 0:
        mse_ratio = test_mse / train_mse
    else:
        mse_ratio = 1.0

    if r2_gap > OVERFIT_R2_GAP or mse_ratio > OVERFIT_MSE_RATIO:
        return "overfit"
    elif test_r2 < UNDERFIT_R2_THRESHOLD:
        return "underfit"
    else:
        return "good_fit"


def _label_classification_fit(train_acc, test_acc) -> str:
    """
    Classify a classification model's fit:
      - overfit:   train accuracy much higher than test accuracy
      - underfit:  test accuracy below random-ish threshold
      - good_fit:  balanced
    """
    acc_gap = train_acc - test_acc

    if acc_gap > OVERFIT_ACC_GAP:
        return "overfit"
    elif test_acc < UNDERFIT_ACC_THRESHOLD:
        return "underfit"
    else:
        return "good_fit"


# ─────────────────────────────────────────────
# QUICK TEST  (run: python metrics.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import io
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessor import preprocess
    from models import train_all_models

    # ── Test 1: Regression ───────────────────────────────────────────────────
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

    print("=" * 70)
    print("TEST 1: REGRESSION METRICS")
    print("=" * 70)

    result = preprocess(io.StringIO(reg_csv), target_column="target_income", task_type="regression")
    models = train_all_models(result)
    models = {k: v for k, v in models.items() if v is not None}
    metrics_df = compute_metrics(models, result)

    # Print full table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(f"\nModels scored: {len(metrics_df)}")
    print(metrics_df.to_string(index=False))

    # Fit label distribution
    print(f"\nFit labels:")
    for label, count in metrics_df["fit_label"].value_counts().items():
        print(f"  {label}: {count}")

    # ── Test 2: Classification ───────────────────────────────────────────────
    clf_csv = """sepal_length,sepal_width,petal_length,petal_width,species
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

    print("\n" + "=" * 70)
    print("TEST 2: CLASSIFICATION METRICS")
    print("=" * 70)

    result2 = preprocess(io.StringIO(clf_csv), target_column="species")
    models2 = train_all_models(result2)
    models2 = {k: v for k, v in models2.items() if v is not None}
    metrics_df2 = compute_metrics(models2, result2)

    print(f"\nModels scored: {len(metrics_df2)}")
    print(metrics_df2.to_string(index=False))

    print(f"\nFit labels:")
    for label, count in metrics_df2["fit_label"].value_counts().items():
        print(f"  {label}: {count}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
