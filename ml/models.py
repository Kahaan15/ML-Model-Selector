"""
models.py
=========
Trains all regression OR classification models depending on the task type
detected by the preprocessor. Returns a dict of fitted sklearn Pipeline objects.

Usage:
    from preprocessor import preprocess
    from models import train_all_models

    result = preprocess("dataset.csv", target_column="price")
    models = train_all_models(result)
    # models = {"Linear": <Pipeline>, "Poly_d2": <Pipeline>, ...}

Regression models:
    Linear, Poly d2–d3, DT depth 1/3/5,
    Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR, KNN k=3/5/7

Classification models:
    Logistic Regression, DT depth 1/3/5,
    Random Forest, AdaBoost, KNN k=3/5/7, SVM linear, SVM RBF,
    Naive Bayes, Gradient Boosting
"""

import warnings
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RANDOM_STATE = 42

# Polynomial feature-explosion guards:
# Skip degree 3 when n_features > this threshold
POLY_D3_MAX_FEATURES = 15


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def train_all_models(result, use_class_weighting: bool = False) -> dict:
    """
    Train all models appropriate for the detected task type.

    Parameters
    ----------
    result : PreprocessResult
        Output of preprocessor.preprocess(). Must have X_train, y_train,
        task_type, and feature_names.

    Returns
    -------
    dict
        {"model_name": fitted_pipeline, ...}
    """
    if result.task_type == "regression":
        return _train_regression_models(result)
    else:
        return _train_classification_models(result, use_class_weighting=use_class_weighting)


def compute_cv_summary(result, metrics_df, models, top_n=3, cv_folds=3) -> list[dict]:
    """
    Compute CV mean/std metrics for top-ranked models.

    Parameters
    ----------
    result : PreprocessResult
    metrics_df : pd.DataFrame
    models : dict[str, estimator]
    top_n : int
    cv_folds : int

    Returns
    -------
    list[dict]
        Summary rows for the UI/API payload.
    """
    if metrics_df is None or metrics_df.empty:
        return []

    cv = _safe_cv_folds(result, requested=cv_folds)
    if cv < 2:
        return []

    ranking_col = "test_r2" if result.task_type == "regression" else "test_f1"
    ranked = metrics_df.sort_values(ranking_col, ascending=False).head(max(1, int(top_n)))

    if result.task_type == "regression":
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
        }
        primary_key = "r2"
    else:
        scoring = {
            "f1": "f1_weighted",
            "accuracy": "accuracy",
        }
        primary_key = "f1"

    rows = []
    for _, metric_row in ranked.iterrows():
        model_name = metric_row["model_name"]
        estimator = models.get(model_name)
        if estimator is None:
            continue

        try:
            cv_result = cross_validate(
                estimator,
                result.X_train,
                result.y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
        except Exception as e:
            print(f"WARNING: CV summary failed for '{model_name}': {e}")
            continue

        if result.task_type == "regression":
            rmse_scores = -cv_result["test_rmse"]
            holdout_primary = float(metric_row["test_r2"])
            rows.append({
                "model_name": model_name,
                "cv_folds": int(cv),
                "cv_primary_metric": "r2",
                "cv_primary_mean": round(float(np.mean(cv_result["test_r2"])), 4),
                "cv_primary_std": round(float(np.std(cv_result["test_r2"])), 4),
                "cv_secondary_metric": "rmse",
                "cv_secondary_mean": round(float(np.mean(rmse_scores)), 4),
                "cv_secondary_std": round(float(np.std(rmse_scores)), 4),
                "holdout_primary": round(holdout_primary, 4),
                "holdout_vs_cv_gap": round(holdout_primary - float(np.mean(cv_result[f"test_{primary_key}"])), 4),
            })
        else:
            holdout_primary = float(metric_row["test_f1"])
            rows.append({
                "model_name": model_name,
                "cv_folds": int(cv),
                "cv_primary_metric": "f1",
                "cv_primary_mean": round(float(np.mean(cv_result["test_f1"])), 4),
                "cv_primary_std": round(float(np.std(cv_result["test_f1"])), 4),
                "cv_secondary_metric": "accuracy",
                "cv_secondary_mean": round(float(np.mean(cv_result["test_accuracy"])), 4),
                "cv_secondary_std": round(float(np.std(cv_result["test_accuracy"])), 4),
                "holdout_primary": round(holdout_primary, 4),
                "holdout_vs_cv_gap": round(holdout_primary - float(np.mean(cv_result[f"test_{primary_key}"])), 4),
            })

    return rows


# ─────────────────────────────────────────────
# REGRESSION MODELS (up to 18)
# ─────────────────────────────────────────────
def _train_regression_models(result) -> dict:
    """Build and fit all regression models."""
    X_train = result.X_train
    y_train = result.y_train
    n_features = X_train.shape[1]

    models = {}

    # ── 1. Linear Regression (baseline) ──────────────────────────────────────
    models["Linear"] = _fit(
        Pipeline([("lr", LinearRegression())]),
        X_train, y_train
    )

    # ── 2–3. Polynomial Regression (degree 2–3) ─────────────────────────────
    poly_degrees = [2, 3]
    for d in poly_degrees:
        # Guard against feature explosion
        if d == 3 and n_features > POLY_D3_MAX_FEATURES:
            continue

        models[f"Poly_d{d}"] = _fit(
            Pipeline([
                ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("lr", LinearRegression()),
            ]),
            X_train, y_train
        )

    # ── Decision Tree Regressor ────────────────────────────────────────
    for depth in [1, 3, 5]:
        models[f"DT_depth{depth}"] = _fit(
            Pipeline([("dt", DecisionTreeRegressor(
                max_depth=depth, random_state=RANDOM_STATE
            ))]),
            X_train, y_train
        )

    # ── Ridge Regression ─────────────────────────────────────────────────
    models["Ridge"] = _fit(
        Pipeline([("ridge", Ridge(alpha=1.0))]),
        X_train, y_train
    )

    # ── Lasso Regression ─────────────────────────────────────────────────
    models["Lasso"] = _fit(
        Pipeline([("lasso", Lasso(alpha=1.0, max_iter=10000))]),
        X_train, y_train
    )

    # ── ElasticNet Regression ────────────────────────────────────────────
    models["ElasticNet"] = _fit(
        Pipeline([("en", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000))]),
        X_train, y_train
    )

    # ── Random Forest Regressor ──────────────────────────────────────────
    models["RandomForest"] = _fit(
        Pipeline([("rf", RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── Gradient Boosting Regressor ────────────────────────────────────────
    models["GradientBoost"] = _fit(
        Pipeline([("gb", GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── SVR (Support Vector Regression) ──────────────────────────────────
    models["SVR"] = _fit(
        Pipeline([("svr", SVR(kernel="rbf"))]),
        X_train, y_train
    )

    # ── KNN Regression ────────────────────────────────────────────────
    for k in [3, 5, 7]:
        # Can't have k > n_samples
        if k > len(X_train):
            continue
        models[f"KNN_k{k}"] = _fit(
            Pipeline([("knn", KNeighborsRegressor(n_neighbors=k))]),
            X_train, y_train
        )

    return models


# ─────────────────────────────────────────────
# CLASSIFICATION MODELS (15)
# ─────────────────────────────────────────────
def _train_classification_models(result, use_class_weighting: bool = False) -> dict:
    """Build and fit all classification models."""
    X_train = result.X_train
    y_train = result.y_train
    class_weight = "balanced" if use_class_weighting else None

    models = {}

    # ── 1. Logistic Regression ───────────────────────────────────────────────
    models["LogisticReg"] = _fit(
        Pipeline([("lr", LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight
        ))]),
        X_train, y_train
    )

    # ── Decision Tree Classifier ────────────────────────────────────────
    for depth in [1, 3, 5]:
        models[f"DT_depth{depth}"] = _fit(
            Pipeline([("dt", DecisionTreeClassifier(
                max_depth=depth, random_state=RANDOM_STATE, class_weight=class_weight
            ))]),
            X_train, y_train
        )

    # ── Random Forest Classifier ──────────────────────────────────────────
    models["RandomForest"] = _fit(
        Pipeline([("rf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight=class_weight
        ))]),
        X_train, y_train
    )

    # ── AdaBoost Classifier ──────────────────────────────────────────
    models["AdaBoost"] = _fit(
        Pipeline([("ab", AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── KNN Classifier ────────────────────────────────────────────────
    for k in [3, 5, 7]:
        if k > len(X_train):
            continue
        models[f"KNN_k{k}"] = _fit(
            Pipeline([("knn", KNeighborsClassifier(n_neighbors=k))]),
            X_train, y_train
        )

    # ── 12. SVM (linear kernel) ──────────────────────────────────────────────
    models["SVM_linear"] = _fit(
        Pipeline([("svm", SVC(
            kernel="linear", probability=True, random_state=RANDOM_STATE, class_weight=class_weight
        ))]),
        X_train, y_train
    )

    # ── 13. SVM (RBF kernel) ─────────────────────────────────────────────────
    models["SVM_rbf"] = _fit(
        Pipeline([("svm", SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight=class_weight
        ))]),
        X_train, y_train
    )

    # ── 14. Naive Bayes ──────────────────────────────────────────────────────
    models["NaiveBayes"] = _fit(
        Pipeline([("nb", GaussianNB())]),
        X_train, y_train
    )

    # ── 15. Gradient Boosting ────────────────────────────────────────────────
    models["GradientBoost"] = _fit(
        Pipeline([("gb", GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    return models


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def _fit(pipeline, X_train, y_train):
    """Fit a pipeline and return it. Wraps for clean error messages."""
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        name = list(pipeline.named_steps.keys())[-1]
        print(f"⚠️  Failed to train '{name}': {e}")
        return None
    return pipeline


def _safe_cv_folds(result, requested: int = 3) -> int:
    """Pick a valid CV fold count for available train data."""
    requested = max(2, int(requested))
    n_samples = len(result.y_train)

    if result.task_type == "classification":
        _, counts = np.unique(result.y_train, return_counts=True)
        min_class = int(counts.min()) if len(counts) else 0
        return min(requested, min_class)

    return min(requested, n_samples)


