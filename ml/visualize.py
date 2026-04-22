"""
visualize.py
============
Generates all charts as base64-encoded PNGs (matplotlib).
Different chart sets for regression vs classification.

Usage:
    from visualize import generate_charts
    charts = generate_charts(result, models, metrics_df, best_model_name)
    # charts = {"model_comparison": "base64string...", ...}

Regression charts (8, when supported):
    model_comparison, train_vs_test, r2_comparison,
    poly_complexity, dt_complexity, knn_complexity, predicted_vs_actual,
    feature_importance

Classification charts (7, when supported):
    model_comparison, train_vs_test, f1_comparison,
    dt_complexity, knn_complexity, confusion_matrix, feature_importance
"""

import io
import re
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


# ─────────────────────────────────────────────
# STYLE CONFIGURATION
# ─────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "#111827",
    "text.color": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "grid.color": "#d1d5db",
    "grid.alpha": 0.6,
    "font.size": 10,
}

COLOR_TRAIN = "#0ea5e9"   # blue
COLOR_TEST = "#ef4444"    # red
COLOR_BEST = "#2563eb"    # strong blue accent
COLOR_BAR = "#ef4444"     # bar chart default
CMAP_MATRIX = "YlOrRd"   # confusion matrix colormap


def _apply_style():
    """Apply dark theme style to matplotlib."""
    plt.rcParams.update(STYLE)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def generate_charts(result, models: dict, metrics_df: pd.DataFrame,
                    best_model_name: str) -> dict:
    """
    Generate all charts for the pipeline output.

    Parameters
    ----------
    result : PreprocessResult
    models : dict of fitted pipelines
    metrics_df : pd.DataFrame from compute_metrics()
    best_model_name : str, name of the recommended model

    Returns
    -------
    dict  {"chart_name": "base64_png_string", ...}
    """
    _apply_style()

    if result.task_type == "regression":
        return _regression_charts(result, models, metrics_df, best_model_name)
    else:
        return _classification_charts(result, models, metrics_df, best_model_name)


# ─────────────────────────────────────────────
# REGRESSION CHARTS
# ─────────────────────────────────────────────
def _regression_charts(result, models, df, best_name) -> dict:
    charts = {}

    # 1. Model comparison — test MSE bar chart
    charts["model_comparison"] = _bar_chart(
        names=df["model_name"].tolist(),
        values=df["test_mse"].tolist(),
        title="Test MSE by Model",
        ylabel="Test MSE",
        highlight=best_name,
    )

    # 2. Train vs Test error — grouped bars
    charts["train_vs_test"] = _grouped_bar(
        names=df["model_name"].tolist(),
        values1=df["train_mse"].tolist(),
        values2=df["test_mse"].tolist(),
        label1="Train MSE",
        label2="Test MSE",
        title="Train vs Test MSE (Bias-Variance View)",
        ylabel="MSE",
    )

    # 3. R2 comparison — grouped bars
    charts["r2_comparison"] = _grouped_bar(
        names=df["model_name"].tolist(),
        values1=df["train_r2"].tolist(),
        values2=df["test_r2"].tolist(),
        label1="Train R2",
        label2="Test R2",
        title="Train vs Test R2 Score",
        ylabel="R2 Score",
    )

    # 4. Polynomial complexity curve
    poly_df = df[df["model_name"].str.startswith("Poly_d")]
    if not poly_df.empty:
        degrees = [int(re.search(r"d(\d+)", n).group(1)) for n in poly_df["model_name"]]
        charts["poly_complexity"] = _complexity_curve(
            x_values=degrees,
            train_vals=poly_df["train_mse"].tolist(),
            test_vals=poly_df["test_mse"].tolist(),
            xlabel="Polynomial Degree",
            ylabel="MSE",
            title="Complexity Curve: Polynomial Degree vs Error",
        )

    # 5. Decision Tree complexity curve
    dt_df = df[df["model_name"].str.startswith("DT_depth")]
    if not dt_df.empty:
        depths = []
        for n in dt_df["model_name"]:
            d = n.replace("DT_depth", "")
            depths.append(int(d) if d != "None" else 20)  # plot unlimited as 20
        charts["dt_complexity"] = _complexity_curve(
            x_values=depths,
            train_vals=dt_df["train_mse"].tolist(),
            test_vals=dt_df["test_mse"].tolist(),
            xlabel="Max Depth (20 = unlimited)",
            ylabel="MSE",
            title="Complexity Curve: Decision Tree Depth vs Error",
            best_x=depths[int(np.argmin(dt_df["test_mse"].to_numpy()))],
            best_label="Optimal depth",
        )

    # 6. KNN complexity curve
    knn_df = df[df["model_name"].str.startswith("KNN_k")]
    if not knn_df.empty:
        ks = [int(re.search(r"k(\d+)", n).group(1)) for n in knn_df["model_name"]]
        charts["knn_complexity"] = _complexity_curve(
            x_values=ks,
            train_vals=knn_df["train_mse"].tolist(),
            test_vals=knn_df["test_mse"].tolist(),
            xlabel="k (Number of Neighbors)",
            ylabel="MSE",
            title="Complexity Curve: KNN k-Value vs Error",
            invert_x=True,  # lower k = more complex
        )

    # 7. Predicted vs Actual scatter for best model
    if best_name in models and models[best_name] is not None:
        best_model = models[best_name]
        test_pred = best_model.predict(result.X_test)
        charts["predicted_vs_actual"] = _scatter_pred_vs_actual(
            y_true=result.y_test,
            y_pred=test_pred,
            title=f"Predicted vs Actual ({best_name})",
        )

        feature_importance_chart = _feature_importance_chart(
            best_model,
            result.feature_names,
            title=f"Top Feature Importances ({best_name})",
        )
        if feature_importance_chart is not None:
            charts["feature_importance"] = feature_importance_chart

    return charts


# ─────────────────────────────────────────────
# CLASSIFICATION CHARTS
# ─────────────────────────────────────────────
def _classification_charts(result, models, df, best_name) -> dict:
    charts = {}

    # 1. Model comparison — test accuracy bar chart
    charts["model_comparison"] = _bar_chart(
        names=df["model_name"].tolist(),
        values=df["test_accuracy"].tolist(),
        title="Test Accuracy by Model",
        ylabel="Test Accuracy",
        highlight=best_name,
    )

    # 2. Train vs Test accuracy — grouped bars
    charts["train_vs_test"] = _grouped_bar(
        names=df["model_name"].tolist(),
        values1=df["train_accuracy"].tolist(),
        values2=df["test_accuracy"].tolist(),
        label1="Train Accuracy",
        label2="Test Accuracy",
        title="Train vs Test Accuracy (Bias-Variance View)",
        ylabel="Accuracy",
    )

    # 3. F1 comparison — grouped bars
    charts["f1_comparison"] = _grouped_bar(
        names=df["model_name"].tolist(),
        values1=df["train_f1"].tolist(),
        values2=df["test_f1"].tolist(),
        label1="Train F1",
        label2="Test F1",
        title="Train vs Test F1 Score",
        ylabel="F1 Score (weighted)",
    )

    # 4. Decision Tree complexity curve
    dt_df = df[df["model_name"].str.startswith("DT_depth")]
    if not dt_df.empty:
        depths = []
        for n in dt_df["model_name"]:
            d = n.replace("DT_depth", "")
            depths.append(int(d) if d != "None" else 20)
        charts["dt_complexity"] = _complexity_curve(
            x_values=depths,
            train_vals=dt_df["train_accuracy"].tolist(),
            test_vals=dt_df["test_accuracy"].tolist(),
            xlabel="Max Depth (20 = unlimited)",
            ylabel="Accuracy",
            title="Complexity Curve: Decision Tree Depth vs Accuracy",
            best_x=depths[int(np.argmax(dt_df["test_accuracy"].to_numpy()))],
            best_label="Optimal depth",
        )

    # 5. KNN complexity curve
    knn_df = df[df["model_name"].str.startswith("KNN_k")]
    if not knn_df.empty:
        ks = [int(re.search(r"k(\d+)", n).group(1)) for n in knn_df["model_name"]]
        charts["knn_complexity"] = _complexity_curve(
            x_values=ks,
            train_vals=knn_df["train_accuracy"].tolist(),
            test_vals=knn_df["test_accuracy"].tolist(),
            xlabel="k (Number of Neighbors)",
            ylabel="Accuracy",
            title="Complexity Curve: KNN k-Value vs Accuracy",
            invert_x=True,
        )

    # 6. Confusion matrix for best model
    if best_name in models and models[best_name] is not None:
        best_model = models[best_name]
        test_pred = best_model.predict(result.X_test)
        charts["confusion_matrix"] = _confusion_matrix_chart(
            y_true=result.y_test,
            y_pred=test_pred,
            class_labels=result.class_labels,
            title=f"Confusion Matrix ({best_name})",
        )

        feature_importance_chart = _feature_importance_chart(
            best_model,
            result.feature_names,
            title=f"Top Feature Importances ({best_name})",
        )
        if feature_importance_chart is not None:
            charts["feature_importance"] = feature_importance_chart

    return charts


# ─────────────────────────────────────────────
# CHART PRIMITIVES
# ─────────────────────────────────────────────
def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _bar_chart(names, values, title, ylabel, highlight=None) -> str:
    """Single bar chart with optional highlight on the best model."""
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.0), 5.5))

    colors = [COLOR_BEST if n == highlight else COLOR_BAR for n in names]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="#d1d5db", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--")

    # Value labels on bars — use compact scientific notation for very large numbers
    max_val = max(abs(v) for v in values) if values else 1
    for bar, val in zip(bars, values):
        label = f"{val:.2e}" if max_val > 1e5 else f"{val:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=6.5, color="#111827")

    fig.tight_layout(pad=1.5)
    return _fig_to_base64(fig)


def _grouped_bar(names, values1, values2, label1, label2,
                 title, ylabel) -> str:
    """Grouped bar chart comparing two metrics side by side."""
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.0), 5.5))

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, values1, width, label=label1, color=COLOR_TRAIN,
           edgecolor="#d1d5db", linewidth=0.5)
    ax.bar(x + width / 2, values2, width, label=label2, color=COLOR_TEST,
           edgecolor="#d1d5db", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(axis="y", linestyle="--")

    fig.tight_layout(pad=1.5)
    return _fig_to_base64(fig)


def _complexity_curve(x_values, train_vals, test_vals,
                      xlabel, ylabel, title, invert_x=False,
                      best_x=None, best_label=None) -> str:
    """Line plot showing train vs test error across model complexity."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by x for clean line plot
    sorted_data = sorted(zip(x_values, train_vals, test_vals))
    xs = [d[0] for d in sorted_data]
    trains = [d[1] for d in sorted_data]
    tests = [d[2] for d in sorted_data]

    ax.plot(xs, trains, "o-", color=COLOR_TRAIN, label="Train", linewidth=2, markersize=8)
    ax.plot(xs, tests, "s--", color=COLOR_TEST, label="Test", linewidth=2, markersize=8)

    # Shade the gap (overfitting region)
    ax.fill_between(xs, trains, tests, alpha=0.15, color=COLOR_TEST)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(framealpha=0.8)
    ax.grid(True, linestyle="--")

    if best_x is not None:
        best_indices = [i for i, x in enumerate(xs) if x == best_x]
        if best_indices:
            best_idx = best_indices[0]
            best_y = tests[best_idx]
            ax.axvline(best_x, color=COLOR_BEST, linestyle=":", linewidth=1.6, alpha=0.95)
            label = best_label or "Best point"
            depth_text = "unlimited" if best_x == 20 and "Depth" in title else str(best_x)
            ax.annotate(
                f"{label}: {depth_text}",
                xy=(best_x, best_y),
                xytext=(10, 10),
                textcoords="offset points",
                color="#111827",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=COLOR_BEST, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=COLOR_BEST, linewidth=1.2),
            )

    if invert_x:
        ax.invert_xaxis()

    # Annotate complexity direction
    ax.annotate("More Complex -->", xy=(0.95, 0.02), xycoords="axes fraction",
                ha="right", fontsize=8, color="#888", style="italic")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _scatter_pred_vs_actual(y_true, y_pred, title) -> str:
    """Scatter plot of predicted vs actual values with perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.7, color=COLOR_TRAIN, edgecolors="#d1d5db",
               linewidths=0.5, s=60, zorder=3)

    # Perfect prediction line
    all_vals = np.concatenate([y_true, y_pred])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "--", color=COLOR_TEST, linewidth=1.5, label="Perfect Prediction", zorder=2)

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(framealpha=0.8)
    ax.grid(True, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _feature_importance_chart(model, feature_names, title, top_n=5) -> str | None:
    """Render top feature importances for models that expose them."""
    estimator = _extract_final_estimator(model)
    importances = getattr(estimator, "feature_importances_", None)

    if importances is None or not feature_names:
        return None

    pairs = list(zip(feature_names, importances))
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]
    pairs.reverse()  # horizontal bar chart reads better low-to-high bottom-up

    labels = [name for name, _ in pairs]
    values = [float(score) for _, score in pairs]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(labels, values, color=COLOR_TRAIN, edgecolor="#d1d5db", linewidth=0.6)

    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--")

    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.02 if max(values) > 0 else 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color="#111827",
        )

    fig.tight_layout()
    return _fig_to_base64(fig)


def _extract_final_estimator(model):
    """Get the final estimator from a sklearn Pipeline-like model."""
    if hasattr(model, "named_steps") and model.named_steps:
        return list(model.named_steps.values())[-1]
    return model


def _confusion_matrix_chart(y_true, y_pred, class_labels, title) -> str:
    """Heatmap of the confusion matrix."""
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use class_labels for display if available
    if class_labels:
        display_labels = [str(c) for c in class_labels]
    else:
        display_labels = [str(i) for i in range(cm.shape[0])]

    im = ax.imshow(cm, interpolation="nearest", cmap=CMAP_MATRIX)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Tick labels
    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)

    # Cell value annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "#ffffff" if cm[i, j] > thresh else "#111827"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    return _fig_to_base64(fig)




