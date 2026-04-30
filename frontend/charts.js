/**
 * charts.js
 * =========
 * Handles all frontend logic:
 *   - File upload & drag-and-drop
 *   - /columns API call → populate dropdowns + dataset summary
 *   - /upload API call  → render metrics, charts, recommendation
 *   - Table sorting
 */

// ─── STATE ──────────────────────────────────────────────────────────────────
// Tracks uploaded file and column metadata from the backend
let uploadedFile = null;
let columnData = null;
let currentSortCol = null;
let currentSortDir = "asc";

// ─── DOM REFS ───────────────────────────────────────────────────────────────
const fileInput = document.getElementById("file-input");
const uploadZone = document.getElementById("upload-zone");
const fileInfo = document.getElementById("file-info");
const summaryCard = document.getElementById("summary-card");
const summaryGrid = document.getElementById("summary-grid");
const warningsContainer = document.getElementById("warnings-container");
const configCard = document.getElementById("config-card");
const configValidation = document.getElementById("config-validation");
const targetSelect = document.getElementById("target-select");
const taskSelect = document.getElementById("task-select");
const classWeightSelect = document.getElementById("class-weight-select");
const runBtn = document.getElementById("run-btn");
const resultsSection = document.getElementById("results-section");
const executiveSummaryCard = document.getElementById("executive-summary-card");
const loadingOverlay = document.getElementById("loading-overlay");
const bestModelBadge = document.getElementById("best-model-badge");
const verdictText = document.getElementById("verdict-text");
const cvSummaryCard = document.getElementById("cv-summary-card");
const metricsThead = document.getElementById("metrics-thead");
const metricsTbody = document.getElementById("metrics-tbody");
const fitRulesNote = document.getElementById("fit-rules-note");
const chartsGrid = document.getElementById("charts-grid");
const logContent = document.getElementById("log-content");
const themeToggle = document.getElementById("theme-toggle");
const themeToggleLabel = document.getElementById("theme-toggle-label");
const commandMeta = document.getElementById("command-meta");
const stepUpload = document.getElementById("step-upload");
const stepConfig = document.getElementById("step-config");
const stepResults = document.getElementById("step-results");
const resultsNavButtons = Array.from(document.querySelectorAll(".results-nav-btn"));
const resultsPanels = Array.from(document.querySelectorAll(".results-panel"));


// ─── CHART FRIENDLY NAMES ───────────────────────────────────────────────────
const CHART_NAMES = {
    model_comparison: "Model Comparison",
    train_vs_test: "Train vs Test Error (Bias-Variance)",
    r2_comparison: "R\u00B2 Score Comparison",
    f1_comparison: "F1 Score Comparison",
    poly_complexity: "Polynomial Complexity Curve",
    dt_complexity: "Decision Tree Complexity Curve",
    knn_complexity: "KNN Complexity Curve",
    predicted_vs_actual: "Predicted vs Actual",
    confusion_matrix: "Confusion Matrix",
    feature_importance: "Feature Importance",
};

// Chart display order
const CHART_ORDER = [
    "train_vs_test", "r2_comparison", "f1_comparison",
    "model_comparison", "poly_complexity", "dt_complexity",
    "knn_complexity", "feature_importance", "predicted_vs_actual", "confusion_matrix",
];

const METRIC_TOOLTIPS = {
    test_f1: "F1 balances precision and recall. It is especially useful when class sizes are uneven.",
    test_accuracy: "Accuracy is the fraction of predictions that were correct overall.",
    train_accuracy: "Training accuracy shows how well the model fits the training data.",
    test_precision: "Precision is the share of predicted positives that were actually positive.",
    test_recall: "Recall is the share of actual positives the model successfully found.",
    test_r2: "R² shows how much target variance the model explains. Higher is better.",
    train_r2: "Training R² shows fit on the training set and helps reveal overfitting.",
    test_rmse: "RMSE is the typical prediction error size in the target's original units.",
    test_mse: "MSE penalizes larger errors more strongly than RMSE.",
    train_mse: "Training MSE helps compare fit on seen data versus unseen data.",
};


// ─── THEME ──────────────────────────────────────────────────────────────────
const THEME_STORAGE_KEY = "ml-model-selector-theme";

initTheme();
initResultsWorkspace();
updateWorkflowState("upload");

if (themeToggle) {
    themeToggle.addEventListener("click", toggleTheme);
}

function initTheme() {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = stored || (preferredDark ? "dark" : "light");
    applyTheme(theme);
}

function toggleTheme() {
    const current = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    applyTheme(current === "light" ? "dark" : "light");
}

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);

    if (themeToggleLabel) {
        themeToggleLabel.textContent = theme === "light" ? "Dark mode" : "Light mode";
    }
}

function initResultsWorkspace() {
    if (resultsNavButtons.length === 0 || resultsPanels.length === 0) return;

    resultsNavButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const panelName = btn.dataset.panel;
            if (!panelName) return;
            setActiveResultsPanel(panelName);
        });
    });

    document.addEventListener("keydown", (event) => {
        if (!resultsSection.classList.contains("visible") || event.altKey) return;
        if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;

        const panelMap = {
            "1": "overview",
            "2": "metrics",
            "3": "visuals",
            "4": "preprocessing",
        };

        const panelName = panelMap[event.key];
        if (!panelName) return;
        setActiveResultsPanel(panelName);
    });
}

function setActiveResultsPanel(panelName) {
    resultsNavButtons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.panel === panelName);
    });

    resultsPanels.forEach((panel) => {
        panel.classList.toggle("active", panel.dataset.panel === panelName);
    });
}


// ─── FILE UPLOAD ────────────────────────────────────────────────────────────

fileInput.addEventListener("change", handleFileSelect);

// Drag and drop
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;

    clearConfigValidation();

    // Validate extension
    if (!file.name.toLowerCase().endsWith(".csv")) {
        showError("Only CSV files are supported.");
        setConfigValidation(["Please upload a .csv file to continue."], { file: true });
        return;
    }

    if (file.size === 0) {
        showError("The selected file is empty.");
        setConfigValidation(["The selected CSV is empty. Please choose a file with data rows."], { file: true });
        return;
    }

    uploadedFile = file;
    fileInfo.textContent = `[ OK ] ${file.name} (${formatBytes(file.size)})`;
    fileInfo.style.display = "inline-block";

    // Hide previous results
    resultsSection.classList.remove("visible");
    updateWorkflowState("config");

    // Send to /columns
    fetchColumns(file);
}


// ─── FETCH COLUMNS ──────────────────────────────────────────────────────────

async function fetchColumns(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/columns", { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        columnData = data;
        renderSummary(data);
        populateTargetDropdown(data.columns);
        configCard.style.display = "block";
        configCard.style.animation = "fadeInUp 0.5s ease-out both";
        updateWorkflowState("config");

        // Set auto-detected task type
        if (data.auto_task_type) {
            taskSelect.value = "auto";
        }

    } catch (err) {
        showError("Failed to read file: " + err.message);
    }
}


// ─── RENDER DATASET SUMMARY ────────────────────────────────────────────────

function renderSummary(data) {
    // Warnings
    warningsContainer.innerHTML = "";
    if (data.warnings && data.warnings.length > 0) {
        data.warnings.forEach(w => {
            const div = document.createElement("div");
            div.className = "warning-banner";
            div.textContent = "[ WARN ] " + w;
            warningsContainer.appendChild(div);
        });
    }

    // Stats
    summaryGrid.innerHTML = `
        <div class="stat-item">
            <span class="stat-value">${data.rows.toLocaleString()}</span>
            <div class="stat-label">Rows</div>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.cols}</span>
            <div class="stat-label">Columns</div>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.numeric_count}</span>
            <div class="stat-label">Numeric</div>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.non_numeric_count}</span>
            <div class="stat-label">Non-Numeric</div>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.null_total.toLocaleString()}</span>
            <div class="stat-label">Missing Values</div>
        </div>
        <div class="stat-item">
            <span id="auto-detected-val" class="stat-value" style="color: var(--accent); font-size: 1rem; text-transform: capitalize;">
                ${data.auto_task_type}
            </span>
            <div class="stat-label">Auto-Detected Task</div>
        </div>
    `;

    summaryCard.style.display = "block";
    summaryCard.style.animation = "fadeInUp 0.5s ease-out both";
}


// ─── POPULATE TARGET DROPDOWN ──────────────────────────────────────────────

function populateTargetDropdown(columns) {
    targetSelect.innerHTML = '<option value="">-- select target column --</option>';

    columns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col.name;
        const tag = col.is_numeric ? "num" : "cat";
        const nullTag = col.nulls > 0 ? `, ${col.nulls} nulls` : "";
        opt.textContent = `${col.name} (${tag}, ${col.nunique} unique${nullTag})`;
        targetSelect.appendChild(opt);
    });

    // Pre-select last column (common target position)
    if (columns.length > 0) {
        targetSelect.value = columns[columns.length - 1].name;
        runBtn.disabled = false;
    }

    targetSelect.addEventListener("change", () => {
        clearConfigValidation();
        runBtn.disabled = targetSelect.value === "";
        
        const selectedTarget = columnData?.columns?.find(c => c.name === targetSelect.value);
        if (selectedTarget) {
            let dynAutoTask = "regression";
            if (selectedTarget.dtype === "object" || selectedTarget.nunique <= 15) {
                dynAutoTask = "classification";
            }
            const autoBanner = document.getElementById("auto-detected-val");
            if (autoBanner) autoBanner.textContent = dynAutoTask;
        }
    });
}


// ─── RUN ANALYSIS ──────────────────────────────────────────────────────────

runBtn.addEventListener("click", runAnalysis);
taskSelect.addEventListener("change", clearConfigValidation);
if (classWeightSelect) {
    classWeightSelect.addEventListener("change", clearConfigValidation);
}

async function runAnalysis() {
    const validation = validateRunInputs();
    if (!validation.ok) {
        setConfigValidation(validation.messages, validation.fields);
        return;
    }

    clearConfigValidation();

    // Show loading
    loadingOverlay.classList.add("active");
    runBtn.disabled = true;
    resultsSection.classList.remove("visible");

    const requestStart = performance.now();
    const controller = new AbortController();
    const REQUEST_TIMEOUT_MS = 90000;
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("target_column", targetSelect.value);
    formData.append("task_type", taskSelect.value);
    formData.append("class_weight_mode", classWeightSelect ? classWeightSelect.value : "off");

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: formData,
            signal: controller.signal,
            headers: {
                "Accept": "application/json",
            },
        });

        let data;
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            data = await res.json();
        } else {
            const text = await res.text();
            data = { error: text || `Unexpected response format (${res.status}).` };
        }

        if (!res.ok) {
            const serverError = data?.error || `Server returned ${res.status}.`;
            throw new Error(serverError);
        }

        if (data.error) {
            showError(data.error);
            loadingOverlay.classList.remove("active");
            runBtn.disabled = false;
            return;
        }

        renderResults(data);
        resultsSection.classList.add("visible");
        updateWorkflowState("results");

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 200);

    } catch (err) {
        const elapsedMs = Math.round(performance.now() - requestStart);
        if (err.name === "AbortError") {
            showError(`Pipeline request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s. Please retry; if it repeats, reload once and run again.`);
        } else {
            showError(`Pipeline failed after ${Math.round(elapsedMs / 1000)}s: ${err.message}`);
        }
    } finally {
        clearTimeout(timeoutId);
        loadingOverlay.classList.remove("active");
        runBtn.disabled = false;
    }
}

function validateRunInputs() {
    const messages = [];
    const fields = {};

    if (!uploadedFile) {
        messages.push("Upload a CSV file before running analysis.");
        fields.file = true;
    } else {
        if (!uploadedFile.name.toLowerCase().endsWith(".csv")) {
            messages.push("Only .csv files are accepted.");
            fields.file = true;
        }
        if (uploadedFile.size === 0) {
            messages.push("The uploaded file is empty.");
            fields.file = true;
        }
    }

    if (!columnData || !Array.isArray(columnData.columns) || columnData.columns.length === 0) {
        messages.push("File profiling is incomplete. Wait for the dataset summary to load.");
    }

    if (!targetSelect.value) {
        messages.push("Select a target column to continue.");
        fields.target = true;
    }

    const selectedTask = taskSelect.value;
    if (!["auto", "regression", "classification"].includes(selectedTask)) {
        messages.push("Task type must be Auto-detect, Regression, or Classification.");
        fields.task = true;
    }

    const classWeightMode = classWeightSelect ? classWeightSelect.value : "off";
    if (!["off", "on"].includes(classWeightMode)) {
        messages.push("Class weighting must be Off or On.");
        fields.classWeight = true;
    }

    if (selectedTask === "regression" && classWeightMode === "on") {
        messages.push("Class weighting applies to classification only. Use Off for regression.");
        fields.task = true;
        fields.classWeight = true;
    }

    const selectedTarget = columnData?.columns?.find((col) => col.name === targetSelect.value);
    if (selectedTask === "regression" && selectedTarget && !selectedTarget.is_numeric) {
        messages.push("Regression works best with a numeric target. Pick a numeric target or use Auto-detect.");
        fields.target = true;
        fields.task = true;
    }

    if (selectedTask === "classification" && selectedTarget && Number(selectedTarget.nunique) < 2) {
        messages.push("Classification needs at least 2 unique target values.");
        fields.target = true;
    }

    return {
        ok: messages.length === 0,
        messages,
        fields,
    };
}

function updateWorkflowState(stage) {
    const states = [
        { el: stepUpload, name: "upload" },
        { el: stepConfig, name: "config" },
        { el: stepResults, name: "results" },
    ];

    const activeIndex = states.findIndex((entry) => entry.name === stage);
    states.forEach((entry, index) => {
        if (!entry.el) return;
        entry.el.classList.toggle("active", index === activeIndex);
        entry.el.classList.toggle("done", index < activeIndex);
    });

    const sidebarNav = document.getElementById("sidebar-results-nav");
    if (sidebarNav) {
        sidebarNav.style.display = stage === "results" ? "flex" : "none";
    }
}

function setConfigValidation(messages, fields = {}) {
    if (!configValidation) return;

    if (!messages || messages.length === 0) {
        clearConfigValidation();
        return;
    }

    clearFieldValidationState();
    if (fields.target) targetSelect.classList.add("invalid-field");
    if (fields.task) taskSelect.classList.add("invalid-field");
    if (fields.classWeight && classWeightSelect) classWeightSelect.classList.add("invalid-field");

    configValidation.classList.add("visible");
    configValidation.innerHTML = `<ul>${messages.map((m) => `<li>${escapeHtml(m)}</li>`).join("")}</ul>`;
}

function clearFieldValidationState() {
    targetSelect.classList.remove("invalid-field");
    taskSelect.classList.remove("invalid-field");
    if (classWeightSelect) classWeightSelect.classList.remove("invalid-field");
}

function clearConfigValidation() {
    clearFieldValidationState();
    if (!configValidation) return;
    configValidation.classList.remove("visible");
    configValidation.innerHTML = "";
}


// ─── RENDER RESULTS ─────────────────────────────────────────────────────────

function renderResults(data) {
    setActiveResultsPanel("overview");
    renderCommandBar(data);
    renderExecutiveSummary(data);
    renderRecommendation(data.recommendation, data.charts, data.dataset_info.task_type);
    renderCvSummary(data.cv_summary, data.dataset_info.task_type);
    renderMetricsTable(data.metrics, data.recommendation.best_model, data.dataset_info.task_type);
    renderCharts(data.charts, data.cm_stats);
    renderLog(data.preprocess_log);
}

function renderCommandBar(data) {
    if (!commandMeta) return;

    const datasetInfo = data?.dataset_info || {};
    const recommendation = data?.recommendation || {};
    const originalRows = Number(datasetInfo.original_shape?.[0]);
    const finalRows = Number(datasetInfo.final_shape?.[0]);
    const taskType = String(datasetInfo.task_type || taskSelect.value || "unknown");
    const bestModel = String(recommendation.best_model || "N/A");
    const targetColumn = String(targetSelect?.value || "N/A");
    const classWeightMode = String(datasetInfo.class_weight_mode || classWeightSelect?.value || "off");
    const applied = Boolean(datasetInfo.class_weighting_applied);
    const now = new Date();

    const createChip = (label, value) => {
        return `
            <div class="command-chip">
                <span class="command-chip-label">${escapeHtml(label)}</span>
                <span class="command-chip-value">${escapeHtml(value)}</span>
            </div>
        `;
    };

    const chips = [
        createChip("Dataset", uploadedFile?.name || "current upload"),
        createChip("Task", taskType.toUpperCase()),
        createChip("Target", targetColumn),
        createChip("Class Weight", `${classWeightMode}${applied ? " (applied)" : ""}`),
        createChip("Timestamp", now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })),
    ];

    commandMeta.innerHTML = chips.join("");
}

function renderExecutiveSummary(data) {
    if (!executiveSummaryCard) return;

    const datasetInfo = data?.dataset_info || {};
    const recommendation = data?.recommendation || {};
    const metrics = Array.isArray(data?.metrics) ? data.metrics : [];

    const taskType = String(datasetInfo.task_type || "unknown");
    const bestModel = String(recommendation.best_model || "N/A");

    const originalRows = datasetInfo.original_shape?.[0] ?? null;
    const finalRows = datasetInfo.final_shape?.[0] ?? null;
    const featureCount = Array.isArray(datasetInfo.feature_names) ? datasetInfo.feature_names.length : 0;

    const bestMetricRow = metrics.find((m) => m.model_name === recommendation.best_model) || metrics[0] || null;

    let performanceLine = "Not available for this run.";

    if (taskType === "regression" && bestMetricRow) {
        const testR2 = Number(bestMetricRow.test_r2);
        const testRmse = Number(bestMetricRow.test_rmse);
        performanceLine = `Best test R2: ${formatMetric(testR2)} | RMSE: ${formatMetric(testRmse)}`;
    }

    if (taskType === "classification" && bestMetricRow) {
        const testF1 = Number(bestMetricRow.test_f1);
        const testAcc = Number(bestMetricRow.test_accuracy);
        performanceLine = `Best test F1: ${formatMetric(testF1)} | Accuracy: ${formatMetric(testAcc)}`;
    }

    const overfitCount = Array.isArray(recommendation.overfit_models) ? recommendation.overfit_models.length : 0;
    const underfitCount = Array.isArray(recommendation.underfit_models) ? recommendation.underfit_models.length : 0;
    const totalModels = metrics.length;
    const leakageWarnings = Array.isArray(datasetInfo.leakage_warnings) ? datasetInfo.leakage_warnings : [];
    const classImbalance = datasetInfo.class_imbalance || {};
    const classWeightingApplied = Boolean(datasetInfo.class_weighting_applied);

    let riskLine = "No strong risk flags detected in this run.";
    if (leakageWarnings.length > 0) {
        riskLine = `Leakage risk: ${leakageWarnings.length} warning(s). ${leakageWarnings[0]}`;
    } else if (taskType === "classification" && classImbalance.is_imbalanced) {
        const ratio = Number(classImbalance.imbalance_ratio || 0).toFixed(2);
        riskLine = classWeightingApplied
            ? `Class imbalance handled: ratio ${ratio}:1, weighted training applied.`
            : `Class imbalance risk: ratio ${ratio}:1, consider enabling class weighting.`;
    } else if (totalModels > 0 && overfitCount >= Math.ceil(totalModels * 0.4)) {
        riskLine = `High variance risk: ${overfitCount}/${totalModels} models overfit.`;
    } else if (totalModels > 0 && underfitCount >= Math.ceil(totalModels * 0.4)) {
        riskLine = `High bias risk: ${underfitCount}/${totalModels} models underfit.`;
    } else if (Number.isFinite(finalRows) && finalRows < 200) {
        riskLine = `Small data warning: only ${finalRows} rows after cleaning.`;
    }

    const scopeLine = (Number.isFinite(originalRows) && Number.isFinite(finalRows))
        ? `${originalRows.toLocaleString()} -> ${finalRows.toLocaleString()} rows, ${featureCount} features.`
        : `${featureCount} features used in training.`;

    executiveSummaryCard.innerHTML = `
        <div class="exec-summary-grid">
            <div class="exec-summary-item">
                <div class="exec-summary-label">Target & Scope</div>
                <div class="exec-summary-value">${escapeHtml(taskType.toUpperCase())}</div>
                <div class="exec-summary-sub">${escapeHtml(finalRows || "?")} rows, ${escapeHtml(featureCount)} features trained</div>
            </div>
            <div class="exec-summary-item">
                <div class="exec-summary-label">Winner AI Model</div>
                <div class="exec-summary-value">${escapeHtml(bestModel)}</div>
                <div class="exec-summary-sub">Optimum tradeoff balance</div>
            </div>
            <div class="exec-summary-item">
                <div class="exec-summary-label">Dataset Quality</div>
                <div class="exec-summary-value" style="text-transform: uppercase;">${escapeHtml(recommendation.dataset_quality || "Unknown")}</div>
                <div class="exec-summary-sub">${escapeHtml(recommendation.dataset_quality_desc || "No quality data available.")}</div>
            </div>
            <div class="exec-summary-item">
                <div class="exec-summary-label">System Health</div>
                <div class="exec-summary-value">${riskLine.includes("No strong") ? "Clear" : "Attention"}</div>
                <div class="exec-summary-sub">${escapeHtml(riskLine)}</div>
            </div>
        </div>
    `;
}





// ─── RECOMMENDATION ────────────────────────────────────────────────────────

function renderRecommendation(rec, charts, taskType) {
    // Emoji removed
    bestModelBadge.innerHTML = escapeHtml(rec.best_model);

    // Format verdict: bold the section headers
    let v = escapeHtml(rec.verdict);
    v = v.replace(/^(BEST MODEL:)/m, "<strong>$1</strong>");
    v = v.replace(/^(WHY:)/m, "<strong>$1</strong>");
    v = v.replace(/^(RANKING CRITERIA:)/m, "<strong>$1</strong>");
    v = v.replace(/^(OVERFIT MODELS.*?:)/m, "<strong>$1</strong>");
    v = v.replace(/^(UNDERFIT MODELS.*?:)/m, "<strong>$1</strong>");
    v = v.replace(/^(SUMMARY:)/m, "<strong>$1</strong>");
    v = v.replace(/(HIGH BIAS)/g, '<span style="color: var(--yellow);">HIGH BIAS</span>');
    v = v.replace(/(HIGH VARIANCE)/g, '<span style="color: var(--accent);">HIGH VARIANCE</span>');
    v = v.replace(/(LOW BIAS)/g, '<span style="color: var(--green);">LOW BIAS</span>');
    v = v.replace(/(LOW VARIANCE)/g, '<span style="color: var(--cyan);">LOW VARIANCE</span>');
    
    verdictText.innerHTML = v;
}

function renderCvSummary(cvSummary, taskType) {
    if (!cvSummaryCard) return;

    const rows = Array.isArray(cvSummary) ? cvSummary : [];
    if (rows.length === 0) {
        cvSummaryCard.style.display = "none";
        cvSummaryCard.innerHTML = "";
        return;
    }

    const primaryLabel = taskType === "regression" ? "Cross-Validated R²" : "Cross-Validated F1 Score";
    const secondaryLabel = taskType === "regression" ? "Cross-Validated RMSE" : "Cross-Validated Accuracy";

    const isPct = taskType === "classification";

    const body = rows.map((row) => {
        const mean1 = Number(row.cv_primary_mean);
        const std1 = Number(row.cv_primary_std);
        const format1 = isPct ? `${(mean1 * 100).toFixed(1)}%` : formatMetric(mean1);
        const formatErr1 = isPct ? `±${(std1 * 100).toFixed(1)}%` : `±${formatMetric(std1)}`;

        const mean2 = Number(row.cv_secondary_mean);
        const std2 = Number(row.cv_secondary_std);
        const format2 = isPct ? `${(mean2 * 100).toFixed(1)}%` : formatMetric(mean2);
        const formatErr2 = isPct ? `±${(std2 * 100).toFixed(1)}%` : `±${formatMetric(std2)}`;

        const holdoutP = Number(row.holdout_primary);
        const formatHoldout = isPct ? `${(holdoutP * 100).toFixed(1)}%` : formatMetric(holdoutP);
        
        const gap = Number(row.holdout_vs_cv_gap);
        const gapStr = isPct ? `${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(2)}%` : formatSignedMetric(gap);
        
        return `
            <tr>
                <td><strong>${escapeHtml(row.model_name)}</strong></td>
                <td>
                    <div style="font-size:1.05rem; font-family:'Outfit', sans-serif;">${format1}</div>
                    <div style="font-size:0.8rem; color:var(--text-muted);">${formatErr1}</div>
                </td>
                <td>
                    <div style="font-size:1.05rem; font-family:'Outfit', sans-serif;">${format2}</div>
                    <div style="font-size:0.8rem; color:var(--text-muted);">${formatErr2}</div>
                </td>
                <td>
                    <div style="font-size:1.05rem; font-family:'Outfit', sans-serif;">${formatHoldout}</div>
                </td>
                <td>
                    <div style="font-size:1.05rem; font-family:'Outfit', sans-serif; color: ${Math.abs(gap) > 0.05 ? 'var(--color-warning)' : 'var(--text-secondary)'};">${gapStr}</div>
                </td>
            </tr>
        `;
    }).join("");

    cvSummaryCard.innerHTML = `
        <div class="cv-summary-title">Cross-Validation Summary (Top Models)</div>
        <div class="cv-summary-table-wrap">
            <table class="cv-summary-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>${primaryLabel}</th>
                        <th>${secondaryLabel}</th>
                        <th>Holdout Primary</th>
                        <th>Holdout - CV Gap</th>
                    </tr>
                </thead>
                <tbody>${body}</tbody>
            </table>
        </div>
        <div class="cv-summary-note">Small gap means the holdout score agrees with cross-validation. Large positive or negative gaps may indicate instability.</div>
    `;
    cvSummaryCard.style.display = "block";
}


// ─── METRICS TABLE ──────────────────────────────────────────────────────────

let metricsData = [];
let metricsBestModel = "";
let metricsTaskType = "";

function renderMetricsTable(metrics, bestModel, taskType) {
    metricsData = metrics;
    metricsBestModel = bestModel;
    metricsTaskType = taskType;
    currentSortCol = null;
    currentSortDir = "asc";

    if (fitRulesNote) {
        fitRulesNote.innerHTML = taskType === "regression"
            ? "<strong>Fit Rules:</strong> Overfit if train-test R<sup>2</sup> gap &gt; 0.10 or test MSE is more than 1.5x train MSE. Underfit if test R<sup>2</sup> &lt; 0.40."
            : "<strong>Fit Rules:</strong> Overfit if train-test accuracy gap &gt; 0.10. Underfit if test accuracy &lt; 0.50.";
    }

    if (metrics.length === 0) return;

    // Determine columns based on task type
    let columns;
    if (taskType === "regression") {
        columns = [
            { key: "model_name", label: "Model" },
            { key: "test_r2", label: "Test R\u00B2" },
            { key: "train_r2", label: "Train R\u00B2" },
            { key: "test_mse", label: "Test MSE" },
            { key: "train_mse", label: "Train MSE" },
            { key: "test_rmse", label: "Test RMSE" },
            { key: "fit_label", label: "Fit" },
        ];
    } else {
        columns = [
            { key: "model_name", label: "Model" },
            { key: "test_f1", label: "Test F1" },
            { key: "test_accuracy", label: "Test Acc" },
            { key: "train_accuracy", label: "Train Acc" },
            { key: "test_precision", label: "Precision" },
            { key: "test_recall", label: "Recall" },
            { key: "fit_label", label: "Fit" },
        ];
    }

    // Header
    metricsThead.innerHTML = "<tr>" + columns.map(c =>
        `<th data-key="${c.key}" onclick="sortTable('${c.key}')">${formatHeaderLabel(c)}</th>`
    ).join("") + "</tr>";

    // Body
    buildTableBody(metrics, columns, bestModel);
}

function buildTableBody(metrics, columns, bestModel) {
    if (!columns) {
        // Reconstruct columns from task type
        if (metricsTaskType === "regression") {
            columns = [
                { key: "model_name" }, { key: "test_r2" }, { key: "train_r2" },
                { key: "test_mse" }, { key: "train_mse" }, { key: "test_rmse" },
                { key: "fit_label" },
            ];
        } else {
            columns = [
                { key: "model_name" }, { key: "test_f1" }, { key: "test_accuracy" },
                { key: "train_accuracy" }, { key: "test_precision" }, { key: "test_recall" },
                { key: "fit_label" },
            ];
        }
    }

    metricsTbody.innerHTML = metrics.map(row => {
        const isBest = row.model_name === bestModel;
        const rowClass = isBest ? 'class="best-row"' : '';

        const cells = columns.map(c => {
            let val = row[c.key];
            if (c.key === "fit_label") {
                const isOverfit = val === "overfit" || val === "highly_overfit" || val === "mildly_overfit";
                const cls = val === "good_fit" ? "fit-good" : isOverfit ? "fit-overfit" : "fit-underfit";
                let label = val === "good_fit" ? "Good Fit" : val === "highly_overfit" ? "High Overfit" : val === "mildly_overfit" ? "Mild Overfit" : "Underfit";
                if (val === "overfit") label = "Overfit";
                return `<td><span class="fit-badge ${cls}">${label}</span></td>`;
            }
            if (c.key === "model_name") {
                const star = isBest ? ' &#11088;' : '';
                return `<td><strong>${val}</strong>${star}</td>`;
            }
            if (typeof val === "number") {
                return `<td>${formatMetric(val)}</td>`;
            }
            return `<td>${val}</td>`;
        }).join("");

        return `<tr ${rowClass}>${cells}</tr>`;
    }).join("");
}

function sortTable(key) {
    if (currentSortCol === key) {
        currentSortDir = currentSortDir === "asc" ? "desc" : "asc";
    } else {
        currentSortCol = key;
        currentSortDir = key === "model_name" || key === "fit_label" ? "asc" : "desc";
    }

    const sorted = [...metricsData].sort((a, b) => {
        let va = a[key], vb = b[key];
        if (typeof va === "string") {
            return currentSortDir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
        }
        return currentSortDir === "asc" ? va - vb : vb - va;
    });

    // Update header classes
    document.querySelectorAll("#metrics-thead th").forEach(th => {
        th.classList.remove("sorted-asc", "sorted-desc");
        if (th.dataset.key === key) {
            th.classList.add(currentSortDir === "asc" ? "sorted-asc" : "sorted-desc");
        }
    });

    buildTableBody(sorted, null, metricsBestModel);
}

function formatHeaderLabel(column) {
    const tooltip = METRIC_TOOLTIPS[column.key];
    if (!tooltip) {
        return `<span class="th-label">${column.label}</span>`;
    }

    const safeTooltip = escapeHtml(tooltip);
    return `
        <span class="th-label">
            <span>${column.label}</span>
            <span class="metric-tip" title="${safeTooltip}" aria-label="${safeTooltip}">?</span>
        </span>
    `;
}


// ─── CHARTS ─────────────────────────────────────────────────────────────────

function renderCharts(charts, cmStats) {
    chartsGrid.innerHTML = "";

    // Render in specified order, skip missing
    const orderedKeys = CHART_ORDER.filter(k => k in charts && k !== "confusion_matrix");
    // Add any keys not in CHART_ORDER
    Object.keys(charts).forEach(k => {
        if (k === "confusion_matrix") return;
        if (!orderedKeys.includes(k)) orderedKeys.push(k);
    });

    if (orderedKeys.length === 0) {
        chartsGrid.innerHTML = `<div class="card"><div class="chart-label">No charts available for this run.</div></div>`;
        return;
    }

    chartsGrid.innerHTML = `
        <div class="chart-controls" style="background: var(--bg-card); padding: 1.5rem; border-radius: var(--radius-md); border: 1px solid var(--border-color); text-align: center;">
            <div style="font-weight: 600; font-size: 1.1rem; color: var(--text-primary); margin-bottom: 0.25rem;">Visualizations</div>
            <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.25rem;">Use the dropdown to inspect different charts for this model.</div>
            <div style="max-width: 400px; margin: 0 auto;">
                <select id="chart-select" class="form-control" style="width: 100%;"></select>
            </div>
        </div>
        <div class="chart-stage" id="chart-stage" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background: var(--bg-card); padding: 2.5rem; border-radius: var(--radius-md); border: 1px solid var(--border-color); width: 100%;"></div>
    `;

    const select = document.getElementById("chart-select");
    const stage = document.getElementById("chart-stage");

    orderedKeys.forEach((key) => {
        const friendlyName = CHART_NAMES[key] || key.replace(/_/g, " ");
        const option = document.createElement("option");
        option.value = key;
        option.textContent = friendlyName;
        select.appendChild(option);
    });

    const CHART_SUBTITLES = {
        model_comparison:    "Compares test error (RMSE) or accuracy across all models. The highlighted bar is the recommended model.",
        train_vs_test:       "Side-by-side train vs. test scores. A large gap between bars indicates overfitting — the model memorized training data but failed to generalize.",
        r2_comparison:       "R\u00b2 score for each model on test data. Closer to 1.0 = excellent fit. Negative values mean the model performs worse than a straight mean prediction.",
        f1_comparison:       "F1 Score comparison across all models. A large gap between Train F1 and Test F1 bars means the model overfit to the training classes.",
        poly_complexity:     "How polynomial regression fits change as degree increases. Watch for test score rising sharply then dropping — that is overfitting starting.",
        dt_complexity:       "Decision tree performance versus tree depth. Very deep trees tend to memorize training data (high train, low test score).",
        knn_complexity:      "KNN performance for different values of k. Very low k overfits; very high k underfits by over-smoothing local patterns.",
        predicted_vs_actual: "Each point is one test sample. Points on the diagonal = perfect predictions. Points far from it = large prediction errors.",
        feature_importance:  "How much each feature contributed to the best model's predictions. Higher bar = more influential. Features near zero add almost no value.",
        confusion_matrix:    "Shows how often each actual class was predicted correctly (diagonal) vs. confused with another class (off-diagonal). Higher diagonal = better.",
        roc_curve:           "Trade-off between true positive rate and false positive rate. Closer to the top-left corner = stronger classifier.",
    };

    const renderSelectedChart = (key) => {
        const friendlyName = CHART_NAMES[key] || key.replace(/_/g, " ");
        let insightHtml = "";

        if (key === "confusion_matrix" && cmStats && cmStats.insights && cmStats.insights.length > 0) {
            const bullets = cmStats.insights.map(s => `<li style="margin-bottom:0.4rem;">${escapeHtml(s)}</li>`).join("");
            insightHtml = `<ul class="cm-insights" style="margin-top: 1.5rem; color: var(--text-secondary); text-align: left; background: var(--bg-base); padding: 1rem 1.5rem; border-radius: var(--radius-sm);  list-style: disc inside;">${bullets}</ul>`;
        }

        const subtitle = CHART_SUBTITLES[key];
        const subtitleHtml = subtitle
            ? `<p style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted); text-align: center; max-width: 700px; line-height: 1.5;">${escapeHtml(subtitle)}</p>`
            : "";

        stage.innerHTML = `
            <img src="data:image/png;base64,${charts[key]}" alt="${escapeHtml(friendlyName)}" style="max-width: 100%; height: auto; border: 1px solid var(--border-color); box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: var(--radius-sm);">
            ${subtitleHtml}
            ${insightHtml}
        `;
    };

    select.addEventListener("change", () => renderSelectedChart(select.value));
    renderSelectedChart(orderedKeys[0]);
}

// ─── PREPROCESSING LOG ──────────────────────────────────────────────────────

function renderLog(log) {
    const groups = groupLogEntries(log);

    logContent.innerHTML = `
        <div class="log-groups">
            ${groups.map((group, index) => `
                <div class="log-group">
                    <button
                        class="log-group-header"
                        type="button"
                        onclick="toggleLogGroup(${index})"
                    >
                        <span>${escapeHtml(group.title)}</span>
                        <span class="log-group-meta">
                            <span class="log-group-count">${group.lines.length} step${group.lines.length === 1 ? "" : "s"}</span>
                            <span class="log-group-arrow">&#9660;</span>
                        </span>
                    </button>
                    <div class="log-group-body" id="log-group-body-${index}">
                        ${group.lines.length > 0
                            ? group.lines.map(line => {
                                // Strip out emojis
                                let clean = line.replace(/[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF]/g, '').trim();
                                // Clean up array representations
                                clean = clean.replace(/\[|\]|'/g, '');
                                return `<div class="log-line" style="display:flex; align-items:start; gap:0.5rem; margin-bottom: 0.5rem;">
                                    <span style="color:var(--text-muted);">&bull;</span>
                                    <span>${escapeHtml(clean)}</span>
                                </div>`;
                              }).join("")
                            : '<div class="log-line">No steps captured in this stage for this run.</div>'}
                    </div>
                </div>
            `).join("")}
        </div>
    `;
}

function toggleLogGroup(index) {
    const body = document.getElementById(`log-group-body-${index}`);
    const header = body?.previousElementSibling;
    if (!body || !header) return;

    body.classList.toggle("open");
    header.classList.toggle("open");
}

function groupLogEntries(log) {
    const buckets = [
        { title: "Data Cleaning", lines: [] },
        { title: "Encoding", lines: [] },
        { title: "Missing Values", lines: [] },
        { title: "Split & Scaling", lines: [] },
        { title: "Other", lines: [] },
    ];

    log.forEach(line => {
        const normalized = line.toLowerCase();

        if (matchesAny(normalized, [
            "loaded dataset",
            "renamed",
            "dropped id-like",
            "high-cardinality",
            "duplicate",
            "near-zero variance",
            "highly correlated",
            "outlier",
            "task type",
            "target    :",
            "features  :",
            "final dataset",
            "sanity check passed",
        ])) {
            buckets[0].lines.push(line);
            return;
        }

        if (matchesAny(normalized, [
            "label-encoded",
            "force-encoded",
            "target encoded",
            "classes found",
        ])) {
            buckets[1].lines.push(line);
            return;
        }

        if (matchesAny(normalized, [
            "missing",
            "imputed",
        ])) {
            buckets[2].lines.push(line);
            return;
        }

        if (matchesAny(normalized, [
            "train/test split",
            "standardised",
        ])) {
            buckets[3].lines.push(line);
            return;
        }

        buckets[4].lines.push(line);
    });

    return buckets;
}

function matchesAny(text, patterns) {
    return patterns.some(pattern => text.includes(pattern));
}


// ─── HELPERS ────────────────────────────────────────────────────────────────

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

function formatMetric(val) {
    if (Math.abs(val) >= 1e6) return val.toExponential(2);
    if (Math.abs(val) >= 100) return val.toFixed(1);
    if (Math.abs(val) >= 1) return val.toFixed(4);
    return val.toFixed(4);
}

function formatSignedMetric(val) {
    const sign = val >= 0 ? "+" : "-";
    return `${sign}${formatMetric(Math.abs(val))}`;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function showError(msg) {
    // Show as a temporary error banner at top of container
    const existing = document.querySelector(".error-banner.global-error");
    if (existing) existing.remove();

    const banner = document.createElement("div");
    banner.className = "error-banner global-error";
    banner.textContent = "[ ERROR ] " + msg;
    banner.style.animation = "fadeInDown 0.3s ease-out";

    const container = document.querySelector(".app-container");
    if(container) container.insertBefore(banner, container.children[0]);

    setTimeout(() => {
        banner.style.opacity = "0";
        banner.style.transition = "opacity 0.5s";
        setTimeout(() => banner.remove(), 500);
    }, 6000);
}
