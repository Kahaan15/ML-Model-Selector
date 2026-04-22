"""
app.py
======
Flask backend — thin wrapper around the ML pipeline.
Zero ML logic here. Just HTTP handling, file validation, and JSON responses.

Endpoints:
    GET  /              — serves frontend/index.html
    GET  /health        — {"status": "ok"}
    POST /columns       — accepts CSV, returns column names + auto-detected task type
    POST /upload        — accepts CSV + target_column + task_type, runs full pipeline

Run:
    python app.py
    → http://localhost:5000
"""

import os
import sys
import json
import tempfile
import traceback
import tempfile

from flask import Flask, request, jsonify, send_from_directory

# Add ml/ directory to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))
from pipeline import run_pipeline
from preprocessor import preprocess

import pandas as pd

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")

# Constraints
MIN_ROWS = 100
MAX_ROWS = 50000
MAX_FEATURES = 20
ALLOWED_EXTENSIONS = {".csv"}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("frontend", "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "ok"})


@app.route("/columns", methods=["POST"])
def get_columns():
    """
    Accept a CSV file, return column info + auto-detected task type.
    Used by the frontend to populate the target dropdown and show dataset summary.
    """
    try:
        # ── Validate file ────────────────────────────────────────────────────
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Only CSV files are supported. Got: {ext}"}), 400

        # ── Read CSV ─────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

        # ── Validate size ────────────────────────────────────────────────────
        n_rows, n_cols = df.shape
        warnings = []

        if n_rows < MIN_ROWS:
            warnings.append(f"Dataset has only {n_rows} rows (recommended: {MIN_ROWS}+). Results may be unreliable.")
        if n_rows > MAX_ROWS:
            return jsonify({"error": f"Dataset too large: {n_rows} rows (max: {MAX_ROWS})."}), 400

        # ── Column info ──────────────────────────────────────────────────────
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        if len(numeric_cols) > MAX_FEATURES:
            warnings.append(f"Many numeric columns ({len(numeric_cols)}). Only top {MAX_FEATURES} will be kept.")

        # ── Auto-detect task type for each possible target ───────────────────
        # Simple heuristic: if last column is float with many unique values -> regression
        last_col = df.columns[-1]
        auto_task = "regression"
        if df[last_col].dtype == "object" or df[last_col].nunique() <= 15:
            auto_task = "classification"

        # ── Build column details ─────────────────────────────────────────────
        columns = []
        for col in df.columns:
            columns.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "nunique": int(df[col].nunique()),
                "nulls": int(df[col].isnull().sum()),
                "is_numeric": col in numeric_cols,
            })

        return jsonify({
            "filename": file.filename,
            "rows": n_rows,
            "cols": n_cols,
            "columns": columns,
            "numeric_count": len(numeric_cols),
            "non_numeric_count": len(non_numeric_cols),
            "null_total": int(df.isnull().sum().sum()),
            "auto_task_type": auto_task,
            "warnings": warnings,
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a CSV + target_column + task_type, run full pipeline, return JSON.
    """
    try:
        # ── Validate file ────────────────────────────────────────────────────
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Only CSV files are supported. Got: {ext}"}), 400

        # ── Read parameters ──────────────────────────────────────────────────
        target_column = request.form.get("target_column", None)
        task_type = request.form.get("task_type", None)
        class_weight_mode = (request.form.get("class_weight_mode", "off") or "off").lower()

        if class_weight_mode not in {"off", "on"}:
            return jsonify({"error": "class_weight_mode must be one of: off, on."}), 400

        # "auto" means let preprocessor decide
        if task_type == "auto" or task_type == "":
            task_type = None

        # ── Save to temp file (so preprocessor can open by path) ─────────────
        tmp_dir = tempfile.gettempdir()
        suffix = os.path.splitext(file.filename)[1].lower() or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp_file:
            tmp_path = tmp_file.name
        file.save(tmp_path)

        try:
            # ── Row count validation ─────────────────────────────────────────
            df_check = pd.read_csv(tmp_path, nrows=MAX_ROWS + 1)
            if len(df_check) > MAX_ROWS:
                return jsonify({"error": f"Dataset too large: >{MAX_ROWS} rows."}), 400

            # ── Run pipeline ─────────────────────────────────────────────────
            result = run_pipeline(
                tmp_path,
                target_column=target_column,
                task_type=task_type,
                class_weight_mode=class_weight_mode,
            )

            return jsonify(result)

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ML Bias-Variance Tradeoff System")
    print("=" * 50)
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/health")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
