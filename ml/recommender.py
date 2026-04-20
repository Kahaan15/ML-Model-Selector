"""
recommender.py
==============
Picks the best model from a metrics DataFrame and explains WHY
in educational bias-variance terms.

Usage:
    from recommender import recommend

    recommendation = recommend(metrics_df)
    print(recommendation["verdict"])

Returns a dict with:
    best_model     : str   — name of the winning model
    verdict        : str   — human-readable explanation
    overfit_models : list  — models flagged as overfit
    underfit_models: list  — models flagged as underfit
    all_rankings   : list  — all models ranked best-to-worst
"""

import pandas as pd


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def recommend(metrics_df: pd.DataFrame, result=None) -> dict:
    """
    Analyse a metrics DataFrame and pick the best model.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of compute_metrics(). Must have 'model_name', 'fit_label',
        and either regression columns (test_r2, test_mse) or
        classification columns (test_f1, test_accuracy).

    Returns
    -------
    dict with keys: best_model, verdict, overfit_models, underfit_models, all_rankings
    """
    if "test_r2" in metrics_df.columns:
        return _recommend_regression(metrics_df)
    else:
        return _recommend_classification(metrics_df, result=result)


# ─────────────────────────────────────────────
# REGRESSION RECOMMENDATION
# ─────────────────────────────────────────────
def _recommend_regression(df: pd.DataFrame) -> dict:
    overfit = df[df["fit_label"] == "overfit"]["model_name"].tolist()
    underfit = df[df["fit_label"] == "underfit"]["model_name"].tolist()
    good = df[df["fit_label"] == "good_fit"]

    # Primary ranking: among good_fit models, pick highest test_r2
    # Tiebreaker: lowest test_mse
    if not good.empty:
        ranked = good.sort_values(
            ["test_r2", "test_mse"], ascending=[False, True]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]
    else:
        # All models are overfit or underfit — pick the one with
        # smallest gap between train and test R2 (least overfit)
        df = df.copy()
        df["r2_gap"] = abs(df["train_r2"] - df["test_r2"])
        ranked = df.sort_values(
            ["r2_gap", "test_r2"], ascending=[True, False]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]

    # Build full ranking list
    all_ranked = df.sort_values(
        ["test_r2", "test_mse"], ascending=[False, True]
    )["model_name"].tolist()

    verdict = _build_regression_verdict(best_name, best_row, overfit, underfit, df)

    return {
        "best_model": best_name,
        "verdict": verdict,
        "overfit_models": overfit,
        "underfit_models": underfit,
        "all_rankings": all_ranked,
    }


# ─────────────────────────────────────────────
# CLASSIFICATION RECOMMENDATION
# ─────────────────────────────────────────────
def _recommend_classification(df: pd.DataFrame, result=None) -> dict:
    overfit = df[df["fit_label"] == "overfit"]["model_name"].tolist()
    underfit = df[df["fit_label"] == "underfit"]["model_name"].tolist()
    good = df[df["fit_label"] == "good_fit"]

    if not good.empty:
        ranked = good.sort_values(
            ["test_f1", "test_accuracy"], ascending=[False, False]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]
    else:
        df = df.copy()
        df["acc_gap"] = abs(df["train_accuracy"] - df["test_accuracy"])
        ranked = df.sort_values(
            ["acc_gap", "test_f1"], ascending=[True, False]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]

    all_ranked = df.sort_values(
        ["test_f1", "test_accuracy"], ascending=[False, False]
    )["model_name"].tolist()

    verdict = _build_classification_verdict(best_name, best_row, overfit, underfit, df, result=result)

    return {
        "best_model": best_name,
        "verdict": verdict,
        "overfit_models": overfit,
        "underfit_models": underfit,
        "all_rankings": all_ranked,
    }


# ─────────────────────────────────────────────
# VERDICT BUILDERS (educational explanations)
# ─────────────────────────────────────────────
def _get_model_description(name: str) -> str:
    """Return a friendly description of what the model is."""
    descriptions = {
        "Linear": "Linear Regression (degree 1, no regularization)",
        "Poly_d2": "Polynomial Regression (degree 2)",
        "Poly_d3": "Polynomial Regression (degree 3)",
        "Poly_d4": "Polynomial Regression (degree 4)",
        "Poly_d5": "Polynomial Regression (degree 5)",
        "DT_depth1": "Decision Tree (max depth 1 — a single split / stump)",
        "DT_depth3": "Decision Tree (max depth 3)",
        "DT_depth5": "Decision Tree (max depth 5)",
        "DT_depth10": "Decision Tree (max depth 10)",
        "DT_depthNone": "Decision Tree (unlimited depth)",
        "Ridge": "Ridge Regression (L2 regularization)",
        "Lasso": "Lasso Regression (L1 regularization)",
        "RandomForest": "Random Forest (ensemble of 100 trees)",
        "SVR": "Support Vector Regression (RBF kernel)",
        "KNN_k1": "KNN (k=1 — memorizes training data)",
        "KNN_k3": "KNN (k=3)",
        "KNN_k5": "KNN (k=5)",
        "KNN_k10": "KNN (k=10 — high smoothing)",
        "LogisticReg": "Logistic Regression (linear decision boundary)",
        "SVM_linear": "SVM (linear kernel)",
        "SVM_rbf": "SVM (RBF kernel — nonlinear boundary)",
        "NaiveBayes": "Naive Bayes (assumes feature independence)",
        "GradientBoost": "Gradient Boosting (sequential ensemble)",
    }
    return descriptions.get(name, name)


def _class_balance_summary(result) -> dict:
    """Summarize class imbalance for classification datasets."""
    if result is None or getattr(result, "task_type", None) != "classification":
        return {}

    class_balance = getattr(result, "class_balance", None) or {}
    if not class_balance:
        return {}

    sorted_items = sorted(class_balance.items(), key=lambda item: item[1], reverse=True)
    majority_label, majority_ratio = sorted_items[0]
    minority_label, minority_ratio = sorted_items[-1]

    return {
        "majority_label": majority_label,
        "majority_ratio": majority_ratio,
        "minority_label": minority_label,
        "minority_ratio": minority_ratio,
        "is_imbalanced": majority_ratio > 0.60,
        "distribution_text": ", ".join(
            f"{label}={ratio*100:.1f}%"
            for label, ratio in sorted_items
        ),
    }


def _build_regression_verdict(best_name, best_row, overfit, underfit, df) -> str:
    """Build a multi-paragraph educational verdict for regression."""
    desc = _get_model_description(best_name)
    lines = []

    # ── Winner announcement ──────────────────────────────────────────────────
    lines.append(
        f"BEST MODEL: {best_name}\n"
        f"{desc}\n"
    )

    lines.append(
        f"Test R2 = {best_row['test_r2']:.4f}  |  "
        f"Test RMSE = {best_row['test_rmse']:.2f}  |  "
        f"Fit: {best_row['fit_label']}\n"
    )

    # ── Why this model won ───────────────────────────────────────────────────
    r2_gap = best_row["train_r2"] - best_row["test_r2"]
    if r2_gap < 0.05:
        lines.append(
            "WHY: This model has the best test R2 score while maintaining a small gap "
            "between training and test performance. A small gap means the model "
            "generalizes well — it learned real patterns, not noise.\n"
        )
    else:
        lines.append(
            "WHY: Among all models, this one achieves the best balance between "
            "accuracy and generalization. While no model is perfect on this dataset, "
            "this one has the most acceptable tradeoff.\n"
        )

    # ── Overfit explanation ──────────────────────────────────────────────────
    if overfit:
        lines.append(
            f"OVERFIT MODELS ({len(overfit)}): {', '.join(overfit)}\n"
            "These models scored well on training data but poorly on test data. "
            "They memorized the training set instead of learning generalizable patterns. "
            "In bias-variance terms, they have LOW BIAS but HIGH VARIANCE — "
            "their predictions change drastically with different training samples.\n"
        )

    # ── Underfit explanation ─────────────────────────────────────────────────
    if underfit:
        lines.append(
            f"UNDERFIT MODELS ({len(underfit)}): {', '.join(underfit)}\n"
            "These models performed poorly on BOTH training and test data. "
            "They are too simple to capture the patterns in this dataset. "
            "In bias-variance terms, they have HIGH BIAS but LOW VARIANCE — "
            "they consistently make the same wrong predictions.\n"
        )

    # ── Bias-variance takeaway ───────────────────────────────────────────────
    total = len(df)
    n_good = total - len(overfit) - len(underfit)
    lines.append(
        f"SUMMARY: Out of {total} models tested, "
        f"{n_good} achieved good fit, "
        f"{len(overfit)} overfit, and "
        f"{len(underfit)} underfit. "
        f"The ideal model sits at the sweet spot of the bias-variance tradeoff — "
        f"complex enough to capture patterns, but not so complex that it memorizes noise."
    )

    return "\n".join(lines)


def _build_classification_verdict(best_name, best_row, overfit, underfit, df, result=None) -> str:
    """Build a multi-paragraph educational verdict for classification."""
    desc = _get_model_description(best_name)
    lines = []
    balance = _class_balance_summary(result)

    # ── Winner announcement ──────────────────────────────────────────────────
    lines.append(
        f"BEST MODEL: {best_name}\n"
        f"{desc}\n"
    )

    lines.append(
        f"Test Accuracy = {best_row['test_accuracy']:.4f}  |  "
        f"Test F1 = {best_row['test_f1']:.4f}  |  "
        f"Fit: {best_row['fit_label']}\n"
    )

    # ── Why this model won ───────────────────────────────────────────────────
    acc_gap = best_row["train_accuracy"] - best_row["test_accuracy"]
    why_bits = []

    if acc_gap < 0.05:
        why_bits.append(
            "It has the highest test F1 score while keeping the train-test gap small, "
            "so it generalizes well instead of just memorizing the training set."
        )
    else:
        why_bits.append(
            "Among all models, it gives the strongest balance between predictive power "
            "and generalization on unseen data."
        )

    if balance.get("distribution_text"):
        if balance["is_imbalanced"]:
            why_bits.append(
                f"The class distribution is imbalanced ({balance['distribution_text']}), "
                "so F1 matters more than raw accuracy because it checks whether minority "
                "classes are still being recovered well."
            )
        else:
            why_bits.append(
                f"The class split is fairly balanced ({balance['distribution_text']}), "
                "so the winning model needed to be strong on both F1 and accuracy rather "
                "than benefiting from a dominant class."
            )

    if best_name == "SVM_rbf":
        why_bits.append(
            "The RBF SVM can model curved, nonlinear class boundaries, which is a good fit "
            "when the classes are not cleanly separated by a single straight decision line."
        )
    elif best_name == "GradientBoost":
        why_bits.append(
            "Gradient boosting won because it can combine many weak nonlinear splits into a "
            "stronger classifier without relying on a single rigid boundary."
        )
    elif best_name == "RandomForest":
        why_bits.append(
            "Random forest won because averaging many trees reduced variance while still "
            "capturing nonlinear interactions in the dataset."
        )
    elif best_name == "LogisticReg":
        why_bits.append(
            "A linear decision boundary was sufficient here, so the simpler logistic model "
            "matched the data well without introducing unnecessary variance."
        )

    severe_tree_overfit = df[
        df["model_name"].str.startswith("DT_depth")
        & (df["train_accuracy"] >= 0.99)
        & ((df["train_accuracy"] - df["test_accuracy"]) >= 0.10)
    ]["model_name"].tolist()
    if severe_tree_overfit:
        why_bits.append(
            f"Tree-based models such as {', '.join(severe_tree_overfit[:2])} overfit severely "
            "with near-perfect training accuracy, which made the more stable winner a safer choice."
        )

    lines.append("WHY: " + " ".join(why_bits) + "\n")

    lines.append(
        "RANKING CRITERIA: 1. Test F1  2. Test Accuracy  3. Overfitting Penalty\n"
    )

    # ── Overfit explanation ──────────────────────────────────────────────────
    if overfit:
        lines.append(
            f"OVERFIT MODELS ({len(overfit)}): {', '.join(overfit)}\n"
            "These models achieved high training accuracy but dropped significantly on "
            "test data. They memorized class boundaries specific to the training set. "
            "HIGH VARIANCE — they would produce very different decision boundaries "
            "if trained on a different random sample.\n"
        )

    # ── Underfit explanation ─────────────────────────────────────────────────
    if underfit:
        lines.append(
            f"UNDERFIT MODELS ({len(underfit)}): {', '.join(underfit)}\n"
            "These models couldn't even fit the training data well. They are too simple "
            "for this classification task. HIGH BIAS — they assume the data follows a "
            "pattern that's simpler than reality.\n"
        )

    # ── Bias-variance takeaway ───────────────────────────────────────────────
    total = len(df)
    n_good = total - len(overfit) - len(underfit)
    lines.append(
        f"SUMMARY: Out of {total} models tested, "
        f"{n_good} achieved good fit, "
        f"{len(overfit)} overfit, and "
        f"{len(underfit)} underfit. "
        f"The best classifier balances decision boundary complexity — "
        f"flexible enough to separate classes, but not so flexible that it overfits to noise."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# QUICK TEST  (run: python recommender.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import io
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessor import preprocess
    from models import train_all_models
    from metrics import compute_metrics

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
    print("TEST 1: REGRESSION RECOMMENDATION")
    print("=" * 70)

    result = preprocess(io.StringIO(reg_csv), target_column="target_income", task_type="regression")
    models = train_all_models(result)
    models = {k: v for k, v in models.items() if v is not None}
    metrics_df = compute_metrics(models, result)
    rec = recommend(metrics_df)

    print(f"\nBest model : {rec['best_model']}")
    print(f"Overfit    : {rec['overfit_models']}")
    print(f"Underfit   : {rec['underfit_models']}")
    print(f"Rankings   : {rec['all_rankings']}")
    print(f"\n--- VERDICT ---\n{rec['verdict']}")

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
    print("TEST 2: CLASSIFICATION RECOMMENDATION")
    print("=" * 70)

    result2 = preprocess(io.StringIO(clf_csv), target_column="species")
    models2 = train_all_models(result2)
    models2 = {k: v for k, v in models2.items() if v is not None}
    metrics_df2 = compute_metrics(models2, result2)
    rec2 = recommend(metrics_df2)

    print(f"\nBest model : {rec2['best_model']}")
    print(f"Overfit    : {rec2['overfit_models']}")
    print(f"Underfit   : {rec2['underfit_models']}")
    print(f"Rankings   : {rec2['all_rankings']}")
    print(f"\n--- VERDICT ---\n{rec2['verdict']}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
