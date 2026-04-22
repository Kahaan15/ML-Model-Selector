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
        return _recommend_regression(metrics_df, result=result)
    else:
        return _recommend_classification(metrics_df, result=result)


# ─────────────────────────────────────────────
# REGRESSION RECOMMENDATION
# ─────────────────────────────────────────────
def _recommend_regression(df: pd.DataFrame, result=None) -> dict:
    highly_overfit = df[df["fit_label"] == "highly_overfit"]["model_name"].tolist()
    mildly_overfit = df[df["fit_label"] == "mildly_overfit"]["model_name"].tolist()
    overfit = highly_overfit + mildly_overfit
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
        # All models are overfit or underfit. 
        # It's better to recommend the model with the highest test_r2, 
        # even if it's flawed, rather than a model that is completely underfit.
        ranked = df.sort_values(
            ["test_r2", "test_mse"], ascending=[False, True]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]

    # Build full ranking list
    all_ranked = df.sort_values(
        ["test_r2", "test_mse"], ascending=[False, True]
    )["model_name"].tolist()

    verdict = _build_regression_verdict(best_name, best_row, highly_overfit, mildly_overfit, underfit, df, result=result)
    quality_label, quality_desc = _score_dataset_quality(best_row['test_r2'], "R2 score")

    return {
        "best_model": best_name,
        "verdict": verdict,
        "dataset_quality": quality_label,
        "dataset_quality_desc": quality_desc,
        "overfit_models": overfit,
        "underfit_models": underfit,
        "all_rankings": all_ranked,
    }


# ─────────────────────────────────────────────
# CLASSIFICATION RECOMMENDATION
# ─────────────────────────────────────────────
def _recommend_classification(df: pd.DataFrame, result=None) -> dict:
    highly_overfit = df[df["fit_label"] == "highly_overfit"]["model_name"].tolist()
    mildly_overfit = df[df["fit_label"] == "mildly_overfit"]["model_name"].tolist()
    overfit = highly_overfit + mildly_overfit
    underfit = df[df["fit_label"] == "underfit"]["model_name"].tolist()
    good = df[df["fit_label"] == "good_fit"]

    if not good.empty:
        ranked = good.sort_values(
            ["test_f1", "test_accuracy"], ascending=[False, False]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]
    else:
        ranked = df.sort_values(
            ["test_f1", "test_accuracy"], ascending=[False, False]
        ).reset_index(drop=True)
        best_name = ranked.iloc[0]["model_name"]
        best_row = ranked.iloc[0]

    all_ranked = df.sort_values(
        ["test_f1", "test_accuracy"], ascending=[False, False]
    )["model_name"].tolist()

    verdict = _build_classification_verdict(best_name, best_row, highly_overfit, mildly_overfit, underfit, df, result=result)
    quality_label, quality_desc = _score_dataset_quality(best_row['test_f1'], "F1 score")

    return {
        "best_model": best_name,
        "verdict": verdict,
        "dataset_quality": quality_label,
        "dataset_quality_desc": quality_desc,
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


def _score_dataset_quality(best_score: float, metric_name: str) -> tuple[str, str]:
    """Evaluate overall dataset predictive signal."""
    if best_score < 0.30:
        return "POOR", f"Top {metric_name} is very low. Dataset likely lacks predictive features, or relationships are purely noise."
    elif best_score < 0.65:
        return "MODERATE", "There is real signal, but predictions are noisy. Feature engineering might help."
    elif best_score < 0.90:
        return "STRONG", "The dataset possesses highly predictive, clear relationships."
    else:
        return "EXCEPTIONAL", "Performance is nearly perfect. (Be wary of target leakage!)"


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


def _root_cause_regression(df: pd.DataFrame) -> list[str]:
    """
    Rule-based inference engine: scans model scores and fires
    diagnostic insights about WHY the dataset is behaving a certain way.
    """
    insights = []

    def score(name):
        row = df[df["model_name"] == name]
        return float(row["test_r2"].iloc[0]) if not row.empty else None

    def gap(name):
        row = df[df["model_name"] == name]
        if row.empty:
            return None
        return float(row["train_r2"].iloc[0]) - float(row["test_r2"].iloc[0])

    linear_r2  = score("Linear")
    poly2_r2   = score("Poly_d2")
    poly3_r2   = score("Poly_d3")
    gb_r2      = score("GradientBoost")
    rf_r2      = score("RandomForest")
    svr_r2     = score("SVR")
    rf_gap     = gap("RandomForest")
    gb_gap     = gap("GradientBoost")

    best_r2 = float(df["test_r2"].max())

    # Rule 1: All models struggle — dataset likely lacks signal
    if best_r2 < 0.30:
        insights.append(
            "ROOT CAUSE: All models are struggling. The best test R2 is below 0.30, which suggests "
            "the selected features have weak or no correlation with the target. "
            "The dataset is likely missing key predictive columns, or the target contains heavy random noise."
        )

    # Rule 2: Non-linearity detected
    if linear_r2 is not None and gb_r2 is not None and gb_r2 - linear_r2 > 0.15:
        insights.append(
            "ROOT CAUSE: Non-linear relationships detected. "
            f"Gradient Boosting (R2={gb_r2:.3f}) significantly outperformed Linear Regression (R2={linear_r2:.3f}). "
            "This means the relationship between features and the target is curved or interactive — "
            "a straight line cannot capture it."
        )
    elif linear_r2 is not None and poly3_r2 is not None and poly3_r2 - linear_r2 > 0.10:
        insights.append(
            "ROOT CAUSE: Polynomial structure detected. "
            f"Polynomial d3 (R2={poly3_r2:.3f}) noticeably outperformed Linear (R2={linear_r2:.3f}). "
            "The data likely has a curved, non-constant trend."
        )

    # Rule 3: Tree models overfitting badly
    if rf_gap is not None and rf_gap > 0.30:
        insights.append(
            f"ROOT CAUSE: Tree-based models (e.g. RandomForest, R2 gap={rf_gap:.3f}) show very high overfitting. "
            "This often occurs with high-cardinality categorical features or when N_samples is too small "
            "relative to N_features. The model is memorizing rather than learning."
        )
    elif gb_gap is not None and gb_gap > 0.30:
        insights.append(
            f"ROOT CAUSE: GradientBoost shows a large train-test R2 gap ({gb_gap:.3f}). "
            "It may be overfitting to noise in sequential boosting iterations. "
            "Reducing max_depth or n_estimators would likely help."
        )

    # Rule 4: SVR fails completely
    if svr_r2 is not None and svr_r2 < 0:
        insights.append(
            "ROOT CAUSE: SVR completely failed (negative R2). "
            "This typically means the data was not properly scaled, "
            "or the target range is too wide for RBF kernel default parameters to handle."
        )

    # Rule 5: Linear performs on par with everything — likely actually linear
    if (
        linear_r2 is not None and gb_r2 is not None
        and abs(linear_r2 - gb_r2) < 0.05
        and linear_r2 > 0.60
    ):
        insights.append(
            "ROOT CAUSE: Linear structure confirmed. "
            "Linear Regression scored comparably to ensemble models, indicating "
            "the relationships in this dataset are near-linear and well-captured by a simple model."
        )

    return insights


def _root_cause_classification(df: pd.DataFrame) -> list[str]:
    """
    Rule-based inference for classification tasks.
    """
    insights = []

    def score(name):
        row = df[df["model_name"] == name]
        return float(row["test_f1"].iloc[0]) if not row.empty else None

    def gap(name):
        row = df[df["model_name"] == name]
        if row.empty:
            return None
        return float(row["train_accuracy"].iloc[0]) - float(row["test_accuracy"].iloc[0])

    lr_f1      = score("LogisticReg")
    gb_f1      = score("GradientBoost")
    rf_f1      = score("RandomForest")
    nb_f1      = score("NaiveBayes")
    svm_lin_f1 = score("SVM_linear")
    best_f1    = float(df["test_f1"].max())
    rf_gap     = gap("RandomForest")
    gb_gap     = gap("GradientBoost")

    # Rule 1: Overall low performance
    if best_f1 < 0.60:
        insights.append(
            "ROOT CAUSE: Overall low classification performance. "
            f"The best F1 across all models is only {best_f1:.3f}. "
            "This suggests class boundaries are not well-defined, or features do not "
            "strongly separate the target classes."
        )

    # Rule 2: Linear vs nonlinear gap
    if lr_f1 is not None and gb_f1 is not None and gb_f1 - lr_f1 > 0.10:
        insights.append(
            "ROOT CAUSE: Non-linear decision boundaries detected. "
            f"Gradient Boosting (F1={gb_f1:.3f}) significantly outperforms Logistic Regression (F1={lr_f1:.3f}). "
            "Class boundaries are likely curved or contain feature interactions that a linear model cannot capture."
        )
    elif lr_f1 is not None and gb_f1 is not None and abs(lr_f1 - gb_f1) < 0.05 and lr_f1 > 0.70:
        insights.append(
            "ROOT CAUSE: Linear separability confirmed. "
            "Logistic Regression performed comparably to ensemble models, suggesting "
            "the class boundaries in this dataset are relatively linear."
        )

    # Rule 3: Tree overfitting
    if rf_gap is not None and rf_gap > 0.20:
        insights.append(
            f"ROOT CAUSE: RandomForest shows a high train–test accuracy gap ({rf_gap:.3f}). "
            "This is a classic sign of high variance — the forest memorized training examples "
            "but failed to generalize. May indicate noisy labels or high-cardinality features."
        )

    # Rule 4: Naive Bayes competitive
    if nb_f1 is not None and best_f1 > 0 and nb_f1 / best_f1 > 0.90:
        insights.append(
            "ROOT CAUSE: Naive Bayes performs on par with complex models. "
            "This suggests your features may be roughly conditionally independent of each other, "
            "which is exactly the assumption Naive Bayes makes. The dataset may have a clean, "
            "low-interaction structure."
        )

    return insights


def _build_regression_verdict(best_name, best_row, highly_overfit, mildly_overfit, underfit, df, result=None) -> str:
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

    # ── Contextual RMSE Interpretation ─────────────────────────────────────────
    if result is not None:
        import numpy as np
        y_all = np.concatenate([result.y_train, result.y_test])
        y_mean = float(np.mean(np.abs(y_all)))
        if y_mean > 0:
            rmse_pct = (best_row['test_rmse'] / y_mean) * 100
            if rmse_pct < 10:
                quality = "excellent"
            elif rmse_pct < 25:
                quality = "acceptable"
            elif rmse_pct < 50:
                quality = "poor"
            else:
                quality = "very poor"
            lines.append(
                f"PREDICTION ERROR: On average, predictions are off by ~{rmse_pct:.1f}% of the typical target value. "
                f"This is considered {quality} predictive accuracy for a regression model.\n"
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
    if highly_overfit or mildly_overfit:
        if highly_overfit:
            lines.append(
                f"HIGHLY OVERFIT MODELS ({len(highly_overfit)}): {', '.join(highly_overfit)}\n"
                "These models completely memorized the training data but failed catastrophically on the test data. "
                "In bias-variance terms, this is classic high variance.\n"
            )
        if mildly_overfit:
            lines.append(
                f"MILDLY OVERFIT MODELS ({len(mildly_overfit)}): {', '.join(mildly_overfit)}\n"
                "These models saw a slight dropoff in performance when moving to unseen data. "
                "Their variance is slightly elevated, meaning they might be too complex for the problem.\n"
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

    # ── Root Cause Analysis ──────────────────────────────────────────────────
    root_causes = _root_cause_regression(df)
    if root_causes:
        lines.append("ROOT CAUSE ANALYSIS:")
        for rc in root_causes:
            lines.append(rc + "\n")

    # ── Bias-variance takeaway ───────────────────────────────────────────────
    total = len(df)
    n_overfit = len(highly_overfit) + len(mildly_overfit)
    n_good = total - n_overfit - len(underfit)
    lines.append(
        f"SUMMARY: Out of {total} models tested, "
        f"{n_good} achieved good fit, "
        f"{n_overfit} overfit, and "
        f"{len(underfit)} underfit. "
        f"The ideal model sits at the sweet spot of the bias-variance tradeoff — "
        f"complex enough to capture patterns, but not so complex that it memorizes noise."
    )

    return "\n".join(lines)


def _build_classification_verdict(best_name, best_row, highly_overfit, mildly_overfit, underfit, df, result=None) -> str:
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
    if highly_overfit or mildly_overfit:
        if highly_overfit:
            lines.append(
                f"HIGHLY OVERFIT MODELS ({len(highly_overfit)}): {', '.join(highly_overfit)}\n"
                "These models completely memorized the training data but failed catastrophically on the test data. "
                "In bias-variance terms, this is classic high variance.\n"
            )
        if mildly_overfit:
            lines.append(
                f"MILDLY OVERFIT MODELS ({len(mildly_overfit)}): {', '.join(mildly_overfit)}\n"
                "These models saw a slight dropoff in performance when moving to unseen data. "
                "Their variance is slightly elevated, meaning they might be too complex for the problem.\n"
            )

    # ── Underfit explanation ─────────────────────────────────────────────────
    if underfit:
        lines.append(
            f"UNDERFIT MODELS ({len(underfit)}): {', '.join(underfit)}\n"
            "These models couldn't even fit the training data well. They are too simple "
            "for this classification task. HIGH BIAS — they assume the data follows a "
            "pattern that's simpler than reality.\n"
        )

    # ── Root Cause Analysis ──────────────────────────────────────────────────
    root_causes = _root_cause_classification(df)
    if root_causes:
        lines.append("ROOT CAUSE ANALYSIS:")
        for rc in root_causes:
            lines.append(rc + "\n")

    # ── Bias-variance takeaway ───────────────────────────────────────────────
    total = len(df)
    n_overfit = len(highly_overfit) + len(mildly_overfit)
    n_good = total - n_overfit - len(underfit)
    lines.append(
        f"SUMMARY: Out of {total} models tested, "
        f"{n_good} achieved good fit, "
        f"{n_overfit} overfit, and "
        f"{len(underfit)} underfit. "
        f"The best classifier balances decision boundary complexity — "
        f"flexible enough to separate classes, but not so flexible that it overfits to noise."
    )

    return "\n".join(lines)


