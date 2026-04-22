# ML Model Selector & Bias-Variance Analyzer
A automated machine learning diagnostic tool that trains over 30 models from a single CSV upload and provides human-readable explanations of the bias-variance tradeoff.

Built for students, researchers, and data scientists to move beyond "black box" machine learning and understand the root cause of model performance.

## Features
- Automated task detection: Identifies if your dataset requires Regression or Classification.
- Comprehensive Model training: Audits 18 regression models or 15 classification models simultaneously.
- Complexity Analysis: Compares model families across varying degrees of complexity (depth, k-neighbors, etc.).
- Bias-Variance Diagnostics: Automatically flags models that are overfitting or underfitting.
- Root Cause Inference: Rule-based engine that explains WHY the dataset is behaving a certain way.
- Parallel Processing: High-performance training using scikit-learn pipelines.
- Modern HUD Interface: Clean, professional dashboard for real-time interaction.
- Zero Persistence: Processes all data in-memory or through temporary buffers for privacy.

## Requirements
- Windows 10 or 11 (Recommended)
- Python 3.8 or higher
- Modern web browser (Chrome, Edge, or Firefox)
- Internet connection (for initial library installation)

## Setup Guide
### Step 1 — Install Python
Ensure you have Python installed. You can check by running `python --version` in your terminal.

### Step 2 — Install Required Libraries
Open your Command Prompt or Terminal and run the following command to install the machine learning engine and server dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib flask
```

### Step 3 — Download the Codebase
Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/ML-Model-Selector.git
cd ML-Model-Selector
```

## How to Use
### Step 1 — Launch the Application
Run the main application file from your terminal:
```bash
python app.py
```
You will see a message: `Server: http://localhost:5000`.

### Step 2 — Access the Dashboard
1. Open your web browser and go to `http://localhost:5000`.
2. Click the Upload zone to select your CSV file.
3. The system will automatically scan the columns.
4. Select your "Target Column" from the dropdown.
5. Select your "Task Type" (or leave it on Auto).
6. Click "Run Diagnostic Analysis".

### Step 3 — Review the Verdict
Once processing is complete, the dashboard will update with:
1. The Executive Summary identifying the best model.
2. The detailed "Verdict" explaining the bias-variance tradeoff.
3. Interactive charts showing complexity curves and error distributions.
4. A full ranking table of every model tested.

## Dataset Requirements
To ensure accurate diagnostic results, your CSV should follow these properties:
- Format: Standard .csv files only.
- Rows: Between 100 and 50,000 rows.
- Features: Up to 20 numerical or categorical columns.
- Missing Values: Automatically handled, but lower amounts provide better results.

## Project Structure
The project is organized into a modular ML engine and a modern web interface:
- ml/preprocessor.py: Data cleaning and train/test splitting.
- ml/models.py: Core training logic for all 33+ models.
- ml/metrics.py: Performance calculation and fit labeling.
- ml/recommender.py: The expert system for ranking and verdicts.
- ml/visualize.py: Memory-based chart generation.
- app.py: Flask server and API handler.
- frontend/: User interface assets (HTML, JS, CSS).

## Troubleshooting
- "Dataset too large": The system is capped at 50,000 rows to ensure fast browser response. Use a subset of your data if needed.
- "Negative R2 score": This usually means the dataset has very heavy noise or no predictive features — check the "Root Cause Analysis" section for more info.
- "Server not starting": Ensure no other application is using Port 5000.
- "Column not appearing": Reload the page and ensure the CSV header is in the first row of your file.

## Built With
- Scikit-Learn: Machine learning models and preprocessing.
- Matplotlib: Statistical data visualization.
- Flask: Backend API and server.
- Pandas & NumPy: High-performance data manipulation.
- Vanilla JavaScript: Dynamic HUD interface logic.

## Author
Made by a BTech student to simplify machine learning diagnostics and help students master the concept of the bias-variance tradeoff.

## License
MIT License — free to use, modify, and distribute for educational or professional purposes.
