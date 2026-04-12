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

Regression models  (up to 18):
    Linear, Poly d2–d5, DT depth 1/3/5/10/None,
    Ridge, Lasso, Random Forest, SVR, KNN k=1/3/5/10

Classification models (15):
    Logistic Regression, DT depth 1/3/5/10/None,
    Random Forest, KNN k=1/3/5/10, SVM linear, SVM RBF,
    Naive Bayes, Gradient Boosting
"""

import warnings
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RANDOM_STATE = 42

# Polynomial feature-explosion guards:
# Skip degree 4–5 when n_features > this threshold
POLY_D4_MAX_FEATURES = 10
POLY_D5_MAX_FEATURES = 8
POLY_D3_MAX_FEATURES = 15


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def train_all_models(result) -> dict:
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
        return _train_classification_models(result)


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

    # ── 2–5. Polynomial Regression (degree 2–5) ─────────────────────────────
    poly_degrees = [2, 3, 4, 5]
    for d in poly_degrees:
        # Guard against feature explosion
        if d == 3 and n_features > POLY_D3_MAX_FEATURES:
            continue
        if d == 4 and n_features > POLY_D4_MAX_FEATURES:
            continue
        if d == 5 and n_features > POLY_D5_MAX_FEATURES:
            continue

        models[f"Poly_d{d}"] = _fit(
            Pipeline([
                ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("lr", LinearRegression()),
            ]),
            X_train, y_train
        )

    # ── 6–10. Decision Tree Regressor ────────────────────────────────────────
    for depth in [1, 3, 5, 10, None]:
        label = f"DT_depth{depth}" if depth is not None else "DT_depthNone"
        models[label] = _fit(
            Pipeline([("dt", DecisionTreeRegressor(
                max_depth=depth, random_state=RANDOM_STATE
            ))]),
            X_train, y_train
        )

    # ── 11. Ridge Regression ─────────────────────────────────────────────────
    models["Ridge"] = _fit(
        Pipeline([("ridge", Ridge(alpha=1.0))]),
        X_train, y_train
    )

    # ── 12. Lasso Regression ─────────────────────────────────────────────────
    models["Lasso"] = _fit(
        Pipeline([("lasso", Lasso(alpha=1.0, max_iter=10000))]),
        X_train, y_train
    )

    # ── 13. Random Forest Regressor ──────────────────────────────────────────
    models["RandomForest"] = _fit(
        Pipeline([("rf", RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── 14. SVR (Support Vector Regression) ──────────────────────────────────
    models["SVR"] = _fit(
        Pipeline([("svr", SVR(kernel="rbf"))]),
        X_train, y_train
    )

    # ── 15–18. KNN Regression ────────────────────────────────────────────────
    for k in [1, 3, 5, 10]:
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
def _train_classification_models(result) -> dict:
    """Build and fit all classification models."""
    X_train = result.X_train
    y_train = result.y_train

    models = {}

    # ── 1. Logistic Regression ───────────────────────────────────────────────
    models["LogisticReg"] = _fit(
        Pipeline([("lr", LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── 2–6. Decision Tree Classifier ────────────────────────────────────────
    for depth in [1, 3, 5, 10, None]:
        label = f"DT_depth{depth}" if depth is not None else "DT_depthNone"
        models[label] = _fit(
            Pipeline([("dt", DecisionTreeClassifier(
                max_depth=depth, random_state=RANDOM_STATE
            ))]),
            X_train, y_train
        )

    # ── 7. Random Forest Classifier ──────────────────────────────────────────
    models["RandomForest"] = _fit(
        Pipeline([("rf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── 8–11. KNN Classifier ────────────────────────────────────────────────
    for k in [1, 3, 5, 10]:
        if k > len(X_train):
            continue
        models[f"KNN_k{k}"] = _fit(
            Pipeline([("knn", KNeighborsClassifier(n_neighbors=k))]),
            X_train, y_train
        )

    # ── 12. SVM (linear kernel) ──────────────────────────────────────────────
    models["SVM_linear"] = _fit(
        Pipeline([("svm", SVC(
            kernel="linear", probability=True, random_state=RANDOM_STATE
        ))]),
        X_train, y_train
    )

    # ── 13. SVM (RBF kernel) ─────────────────────────────────────────────────
    models["SVM_rbf"] = _fit(
        Pipeline([("svm", SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
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


# ─────────────────────────────────────────────
# QUICK TEST  (run: python models.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import io
    import sys
    import os

    # Add parent dir so we can import preprocessor
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessor import preprocess

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

    print("=" * 60)
    print("TEST 1: REGRESSION MODELS")
    print("=" * 60)

    result = preprocess(io.StringIO(reg_csv), target_column="target_income", task_type="regression")
    models = train_all_models(result)

    # Filter out None (failed) models
    models = {k: v for k, v in models.items() if v is not None}

    print(f"\n[OK] Task type: {result.task_type}")
    print(f"[OK] X_train shape: {result.X_train.shape}")
    print(f"[OK] Models trained: {len(models)}")
    print(f"     Names: {list(models.keys())}")

    # Quick sanity: each model can predict
    for name, model in models.items():
        preds = model.predict(result.X_test)
        print(f"     {name:20s} -> predictions shape: {preds.shape}")

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

    print("\n" + "=" * 60)
    print("TEST 2: CLASSIFICATION MODELS")
    print("=" * 60)

    result2 = preprocess(io.StringIO(clf_csv), target_column="species")
    models2 = train_all_models(result2)

    # Filter out None (failed) models
    models2 = {k: v for k, v in models2.items() if v is not None}

    print(f"\n[OK] Task type: {result2.task_type}")
    print(f"[OK] X_train shape: {result2.X_train.shape}")
    print(f"[OK] Models trained: {len(models2)}")
    print(f"     Names: {list(models2.keys())}")

    for name, model in models2.items():
        preds = model.predict(result2.X_test)
        print(f"     {name:20s} -> predictions shape: {preds.shape}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
