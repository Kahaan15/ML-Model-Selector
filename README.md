# ML Model Selector

> Plug in any CSV — automatically trains 15+ ML models, compares bias-variance tradeoff, and recommends the best model with explanation.

---

## What It Does

Upload any CSV dataset and this system will:

- Auto-detect whether your problem is **regression or classification**
- Train **18 regression models** or **15 classification models** at varying complexities
- Compare **train vs test errors** across all models
- Visualize the **bias-variance tradeoff** through interactive charts
- **Recommend the best model** with a human-readable explanation of why

The goal is educational — understanding *why* a model works, not just *that* it works.

---

## Models Trained

### Regression (18 models)
- Linear Regression (baseline)
- Polynomial Regression — degrees 2, 3, 4, 5
- Decision Tree — depths 1, 3, 5, 10, unlimited
- Ridge, Lasso (regularized linear)
- Random Forest, SVR
- KNN — k = 1, 3, 5, 10

### Classification (15 models)
- Logistic Regression (baseline)
- Decision Tree — depths 1, 3, 5, 10, unlimited
- Random Forest, Gradient Boosting
- SVM — linear and RBF kernels
- Naive Bayes
- KNN — k = 1, 3, 5, 10

---

## Charts Generated

| Chart | Purpose |
|---|---|
| Model comparison | Test error side-by-side for all models |
| Train vs Test error | Visualizes overfitting and underfitting |
| Complexity curves | How error changes with model complexity (degree / depth / k) |
| R² comparison | Goodness of fit across all models |
| Predicted vs Actual | Scatter plot for the best model |
| Confusion matrix | For classification tasks |

---

## Project Structure

```
ml-model-selector/
├── ml/
│   ├── preprocessor.py     # Cleaning, encoding, scaling, train/test split
│   ├── models.py           # All model training
│   ├── metrics.py          # MSE, RMSE, R², Accuracy, F1, Precision, Recall
│   ├── recommender.py      # Best model selection + verdict
│   ├── visualize.py        # All charts as base64 PNGs
│   ├── pipeline.py         # Single function that orchestrates everything
│   └── test_pipeline.py    # End-to-end tests
├── app.py                  # Flask backend
├── frontend/
│   ├── index.html          # Single-page UI
│   └── charts.js           # Fetch, render charts, display results
└── outputs/                # Generated charts (auto-created)
```

---

## Dataset Requirements

| Property | Requirement |
|---|---|
| Format | CSV only |
| Rows | 100 – 50,000 |
| Features | 1 – 20 numerical columns |
| Target | Pick from dropdown after upload |
| Missing values | Handled automatically |

---

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib flask
```

---

## Usage

### Run the full web app
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

### Run ML scripts standalone
```bash
python ml/preprocessor.py
python ml/models.py
python ml/metrics.py
python ml/pipeline.py
```

Each script has a built-in test that runs automatically.

---

## How It Works

```
CSV Upload
    ↓
preprocessor.py   →   cleans, encodes, scales, splits
    ↓
models.py         →   trains 18 or 15 models
    ↓
metrics.py        →   computes MSE, R², F1, fit labels
    ↓
recommender.py    →   ranks models, generates verdict
    ↓
visualize.py      →   generates all charts
    ↓
Flask + Frontend  →   displays everything
```

The Flask backend is a thin wrapper — it imports `pipeline.py` and exposes it as an API endpoint. All ML logic lives entirely in the `ml/` scripts.

---

## Tech Stack

- **ML:** scikit-learn, numpy, pandas
- **Charts:** matplotlib
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript

---

## Status

| Module | Status |
|---|---|
| `preprocessor.py` | ✅ Complete |
| `models.py` | 🔲 In progress |
| `metrics.py` | 🔲 In progress |
| `recommender.py` | 🔲 In progress |
| `visualize.py` | 🔲 In progress |
| `pipeline.py` | 🔲 In progress |
| `app.py` | 🔲 In progress |
| `frontend/` | 🔲 In progress |
