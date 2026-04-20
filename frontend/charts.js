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
const targetSelect = document.getElementById("target-select");
const taskSelect = document.getElementById("task-select");
const runBtn = document.getElementById("run-btn");
const resultsSection = document.getElementById("results-section");
const loadingOverlay = document.getElementById("loading-overlay");
const bestModelBadge = document.getElementById("best-model-badge");
const verdictText = document.getElementById("verdict-text");
const metricsThead = document.getElementById("metrics-thead");
const metricsTbody = document.getElementById("metrics-tbody");
const fitRulesNote = document.getElementById("fit-rules-note");
const chartsGrid = document.getElementById("charts-grid");
const logContent = document.getElementById("log-content");
const themeToggle = document.getElementById("theme-toggle");
const themeToggleLabel = document.getElementById("theme-toggle-label");


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

    // Validate extension
    if (!file.name.toLowerCase().endsWith(".csv")) {
        showError("Only CSV files are supported.");
        return;
    }

    uploadedFile = file;
    fileInfo.textContent = `\u2705 ${file.name} (${formatBytes(file.size)})`;
    fileInfo.style.display = "inline-block";

    // Hide previous results
    resultsSection.classList.remove("visible");

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
            div.textContent = "\u26A0 " + w;
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
            <span class="stat-value" style="color: var(--accent); font-size: 1rem; text-transform: capitalize;">
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
        runBtn.disabled = targetSelect.value === "";
    });
}


// ─── RUN ANALYSIS ──────────────────────────────────────────────────────────

runBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
    if (!uploadedFile || !targetSelect.value) return;

    // Show loading
    loadingOverlay.classList.add("active");
    runBtn.disabled = true;
    resultsSection.classList.remove("visible");

    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("target_column", targetSelect.value);
    formData.append("task_type", taskSelect.value);

    try {
        const res = await fetch("/upload", { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            showError(data.error);
            loadingOverlay.classList.remove("active");
            runBtn.disabled = false;
            return;
        }

        renderResults(data);
        resultsSection.classList.add("visible");

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 200);

    } catch (err) {
        showError("Pipeline failed: " + err.message);
    } finally {
        loadingOverlay.classList.remove("active");
        runBtn.disabled = false;
    }
}


// ─── RENDER RESULTS ─────────────────────────────────────────────────────────

function renderResults(data) {
    renderRecommendation(data.recommendation);
    renderBaseline(data.baseline, data.metrics, data.dataset_info.task_type);
    renderMetricsTable(data.metrics, data.recommendation.best_model, data.dataset_info.task_type);
    renderCharts(data.charts, data.cm_stats);
    renderLog(data.preprocess_log);
}


// ─── BASELINE CALLOUT ───────────────────────────────────────────────────────

function renderBaseline(baseline, metrics, taskType) {
    const container = document.getElementById("baseline-callout");
    if (!container || !baseline) return;

    let statHtml = "";
    let beatCount = 0;
    const total = metrics.length;

    if (taskType === "regression") {
        const baseRmse = baseline.test_rmse;
        beatCount = metrics.filter(m => m.test_rmse < baseRmse).length;
        statHtml = `
            <span class="baseline-stat">Baseline RMSE: <strong>${baseRmse.toFixed(4)}</strong></span>
            <span class="baseline-stat">Baseline R&sup2;: <strong>0.0000</strong></span>
        `;
    } else {
        const baseAcc = baseline.test_accuracy;
        const baseF1 = baseline.test_f1;
        beatCount = metrics.filter(m => m.test_f1 > baseF1).length;
        statHtml = `
            <span class="baseline-stat">Strategy: <strong>${baseline.strategy}</strong></span>
            <span class="baseline-stat">Baseline Acc: <strong>${baseAcc.toFixed(4)}</strong></span>
            <span class="baseline-stat">Baseline F1: <strong>${baseF1.toFixed(4)}</strong></span>
            <span class="baseline-stat" style="color:var(--text-muted);font-size:0.78rem;">${baseline.description}</span>
        `;
    }

    const allBeat = beatCount === total;
    const beatColor = allBeat ? "var(--green)" : "var(--yellow)";
    const beatIcon = allBeat ? "✅" : "⚠️";

    container.innerHTML = `
        <div class="baseline-inner">
            <div class="baseline-left">
                <div class="baseline-title">📊 Naive Baseline</div>
                <div class="baseline-stats">${statHtml}</div>
            </div>
            <div class="baseline-right">
                <div class="baseline-beat" style="color:${beatColor}">
                    ${beatIcon} ${beatCount}/${total} models beat the baseline
                </div>
                ${allBeat ? '<div class="baseline-note">All models outperform random prediction ✔</div>' : ''}
            </div>
        </div>
    `;
    container.style.display = "block";
}


// ─── RECOMMENDATION ────────────────────────────────────────────────────────

function renderRecommendation(rec) {
    bestModelBadge.innerHTML = `&#127942; ${escapeHtml(rec.best_model)}`;

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
                const cls = val === "good_fit" ? "fit-good" : val === "overfit" ? "fit-overfit" : "fit-underfit";
                const label = val === "good_fit" ? "Good" : val === "overfit" ? "Overfit" : "Underfit";
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
    const orderedKeys = CHART_ORDER.filter(k => k in charts);
    // Add any keys not in CHART_ORDER
    Object.keys(charts).forEach(k => {
        if (!orderedKeys.includes(k)) orderedKeys.push(k);
    });

    if (orderedKeys.length === 0) {
        chartsGrid.innerHTML = `<div class="card"><div class="chart-label">No charts available for this run.</div></div>`;
        return;
    }

    chartsGrid.innerHTML = `
        <div class="chart-controls">
            <div class="chart-controls-copy">
                <strong>Focused chart view</strong>
                <span>Select any graph and inspect it at a much larger size.</span>
            </div>
            <div class="chart-select-wrap">
                <select id="chart-select"></select>
            </div>
        </div>
        <div class="chart-stage" id="chart-stage"></div>
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

    const renderSelectedChart = (key) => {
        const friendlyName = CHART_NAMES[key] || key.replace(/_/g, " ");
        let insightHtml = "";
        let stageClass = "chart-stage";

        if (key === "confusion_matrix") {
            stageClass += " chart-stage-matrix";
        } else if (["poly_complexity", "dt_complexity", "knn_complexity"].includes(key)) {
            stageClass += " chart-stage-compact";
        }

        if (key === "confusion_matrix" && cmStats && cmStats.insights && cmStats.insights.length > 0) {
            const bullets = cmStats.insights.map(s => `<li>${escapeHtml(s)}</li>`).join("");
            insightHtml = `<ul class="cm-insights">${bullets}</ul>`;
        }

        stage.className = stageClass;
        stage.innerHTML = `
            <div class="chart-stage-meta">
                <strong>${escapeHtml(friendlyName)}</strong>
                <span>Use the dropdown to switch visualizations</span>
            </div>
            <img src="data:image/png;base64,${charts[key]}" alt="${escapeHtml(friendlyName)}" loading="lazy">
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
                        class="log-group-header ${index === 0 ? "open" : ""}"
                        type="button"
                        onclick="toggleLogGroup(${index})"
                    >
                        <span>${escapeHtml(group.title)}</span>
                        <span class="log-group-meta">
                            <span class="log-group-count">${group.lines.length} step${group.lines.length === 1 ? "" : "s"}</span>
                            <span class="log-group-arrow">&#9660;</span>
                        </span>
                    </button>
                    <div class="log-group-body ${index === 0 ? "open" : ""}" id="log-group-body-${index}">
                        ${group.lines.map(line => `<div class="log-line">${escapeHtml(line)}</div>`).join("")}
                    </div>
                </div>
            `).join("")}
        </div>
    `;
}

function toggleLog() {
    const content = logContent;
    const toggle = document.getElementById("log-toggle");
    content.classList.toggle("open");
    toggle.innerHTML = content.classList.contains("open")
        ? "&#9650; Hide preprocessing steps"
        : "&#9660; Show preprocessing steps";
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

    return buckets.filter(group => group.lines.length > 0);
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
    banner.textContent = "\u274C " + msg;
    banner.style.animation = "fadeInDown 0.3s ease-out";

    const container = document.querySelector(".container");
    container.insertBefore(banner, container.children[1]); // after header

    setTimeout(() => {
        banner.style.opacity = "0";
        banner.style.transition = "opacity 0.5s";
        setTimeout(() => banner.remove(), 500);
    }, 6000);
}
