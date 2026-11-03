document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentArtifacts = null;
    let currentViewMode = "composite";

    // DOM Elements
    const demoList = document.getElementById("demo-list");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const loadingOverlay = document.getElementById("loading-overlay");

    // Viewport Elements
    const rawImage = document.getElementById("raw-image");
    const aiImage = document.getElementById("ai-image");
    const rawDims = document.getElementById("raw-dims");
    const aiLayerTag = document.getElementById("ai-layer-tag");

    // Buttons
    const btnComposite = document.getElementById("btn-composite");
    const btnSeg = document.getElementById("btn-seg");
    const btnDet = document.getElementById("btn-det");

    // Report Elements
    const triageBanner = document.getElementById("triage-banner");
    const triageIcon = document.getElementById("triage-icon");
    const triageTitle = document.getElementById("triage-title");
    const triageSummary = document.getElementById("triage-summary");

    const ctrValue = document.getElementById("ctr-value");
    const ctrStatus = document.getElementById("ctr-status");
    const collapseValue = document.getElementById("collapse-value");
    const collapseStatus = document.getElementById("collapse-status");

    const findingsTableBody = document.getElementById("findings-tbody");
    const latencyInfo = document.getElementById("latency-info");
    const impressionText = document.getElementById("impression-text");

    // 1. Fetch Demo Cases
    async function loadDemoCases() {
        try {
            const res = await fetch("/api/demo-cases");
            const data = await res.json();
            renderDemoCases(data.cases);

            // Automatically run first case
            if (data.cases && data.cases.length > 0) {
                runDemoCase(data.cases[0].filename);
            }
        } catch (err) {
            console.error("Failed to fetch demo cases:", err);
        }
    }

    function renderDemoCases(cases) {
        demoList.innerHTML = "";
        cases.forEach((c, idx) => {
            const card = document.createElement("div");
            card.className = `demo-card ${idx === 0 ? "active" : ""}`;
            card.dataset.filename = c.filename;

            card.innerHTML = `
                <img src="${c.thumbnail_url}" class="demo-thumb" alt="${c.title}" />
                <div class="demo-info">
                    <div class="demo-title">${c.title}</div>
                    <div class="demo-sub">${c.category}</div>
                </div>
            `;

            card.addEventListener("click", () => {
                document.querySelectorAll(".demo-card").forEach(el => el.classList.remove("active"));
                card.classList.add("active");
                runDemoCase(c.filename);
            });

            demoList.appendChild(card);
        });
    }

    // 2. Run Demo Prediction
    async function runDemoCase(filename) {
        showLoading(true);
        try {
            const res = await fetch("/api/predict/demo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename })
            });
            const report = await res.json();
            displayReport(report);
        } catch (err) {
            console.error("Inference failed:", err);
            alert("Inference error: " + err.message);
        } finally {
            showLoading(false);
        }
    }

    // 3. Run Uploaded File Prediction
    async function uploadAndPredict(file) {
        showLoading(true);
        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/predict/upload", {
                method: "POST",
                body: formData
            });
            const report = await res.json();
            // Deselect demo cards
            document.querySelectorAll(".demo-card").forEach(el => el.classList.remove("active"));
            displayReport(report);
        } catch (err) {
            console.error("Upload inference failed:", err);
            alert("Upload error: " + err.message);
        } finally {
            showLoading(false);
        }
    }

    // 4. Update UI with Diagnostic Report
    function displayReport(report) {
        currentArtifacts = report.artifacts;

        // Viewport Images
        rawImage.src = report.artifacts.raw_url;
        rawDims.textContent = `${report.dimensions} | ${report.source_image}`;
        updateAILayerView(currentViewMode);

        // Triage Badge
        const alertInfo = report.triage_alert;
        triageBanner.className = `triage-banner ${alertInfo.badge_class}`;
        triageTitle.textContent = alertInfo.level;
        triageSummary.textContent = alertInfo.summary;

        if (alertInfo.level === "EMERGENCY") {
            triageIcon.textContent = "🚨";
        } else if (alertInfo.level === "PRIORITY") {
            triageIcon.textContent = "⚠️";
        } else {
            triageIcon.textContent = "✅";
        }

        // Biomarkers
        const ctr = report.biomarkers.cardiothoracic_ratio;
        if (ctr && ctr.ctr_value !== null) {
            ctrValue.textContent = ctr.ctr_value.toFixed(2);
            ctrStatus.textContent = ctr.status;
        } else {
            ctrValue.textContent = "---";
            ctrStatus.textContent = "Not Assessed / Normal";
        }

        const ptx = report.biomarkers.pleural_collapse;
        collapseValue.textContent = `${ptx.collapse_percentage}%`;
        collapseStatus.textContent = ptx.severity;

        // Findings Table
        findingsTableBody.innerHTML = "";
        const localized = report.findings.localized_lesions;

        if (ptx.detected) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><span class="class-tag" style="background: rgba(0, 255, 127, 0.2); color: #00ff7f;">Pneumothorax</span></td>
                <td>${(ptx.confidence_mean * 100).toFixed(1)}%</td>
                <td>Visceral Pleura</td>
                <td>${ptx.intervention_recommended}</td>
            `;
            findingsTableBody.appendChild(tr);
        }

        localized.forEach(det => {
            const tr = document.createElement("tr");
            let tagColor = "#f59e0b";
            let tagBg = "rgba(245, 158, 11, 0.2)";

            if (det.class_name.includes("Cardiomegaly")) {
                tagColor = "#ef4444";
                tagBg = "rgba(239, 68, 68, 0.2)";
            } else if (det.class_name.includes("Aortic")) {
                tagColor = "#f97316";
                tagBg = "rgba(249, 115, 22, 0.2)";
            } else if (det.class_name.includes("Pleural")) {
                tagColor = "#38bdf8";
                tagBg = "rgba(56, 189, 248, 0.2)";
            }

            tr.innerHTML = `
                <td><span class="class-tag" style="background: ${tagBg}; color: ${tagColor};">${det.class_name}</span></td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td>${det.lung_zone}</td>
                <td>Radiological Correlation</td>
            `;
            findingsTableBody.appendChild(tr);
        });

        if (!ptx.detected && localized.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--text-dim); padding: 1rem;">No focal lesions or visceral pleural detachment detected.</td>`;
            findingsTableBody.appendChild(tr);
        }

        // Latency
        const lat = report.latency;
        latencyInfo.textContent = `Total: ${lat.total_ms}ms (U-Net++: ${lat.unet_ms}ms | YOLO11m: ${lat.yolo_ms}ms)`;

        // Structured Impression Text
        generateImpression(report);
    }

    function generateImpression(report) {
        const ptx = report.findings.pneumothorax_segmentation;
        const ctr = report.biomarkers.cardiothoracic_ratio;
        const lesions = report.findings.localized_lesions;

        let lines = [];
        lines.push(`[EXAM]: ${report.exam_id} (${report.source_image})`);
        lines.push(`[TRIAGE]: ${report.triage_alert.level} PRIORITY`);
        lines.push(`\nFINDINGS:`);

        if (ptx.detected) {
            lines.push(`- Pleura: Pneumothorax confirmed with visceral line detachment. Lung collapse estimated at ${ptx.collapse_percentage}% (${ptx.severity}). Recommended: ${ptx.intervention_recommended}.`);
        } else {
            lines.push(`- Pleura: No pneumothorax or acute pleural separation delineated.`);
        }

        if (ctr && ctr.ctr_value) {
            lines.push(`- Mediastinum / Heart: CTR = ${ctr.ctr_value} (${ctr.status}).`);
        } else {
            lines.push(`- Mediastinum / Heart: Cardiothoracic silhouette within normal limits.`);
        }

        if (lesions.length > 0) {
            const nonCardiac = lesions.filter(l => !l.class_name.includes("Cardiomegaly"));
            if (nonCardiac.length > 0) {
                const listStr = nonCardiac.map(l => `${l.class_name} in ${l.lung_zone} (conf: ${l.confidence})`).join("; ");
                lines.push(`- Parenchyma / Outlines: ${listStr}.`);
            }
        }

        lines.push(`\nCONCLUSION: Structured diagnostic synthesis automatically formatted per RadLex standards.`);
        impressionText.textContent = lines.join("\n");
    }

    // 5. Layer Switching
    function updateAILayerView(mode) {
        if (!currentArtifacts) return;
        currentViewMode = mode;

        btnComposite.classList.toggle("active", mode === "composite");
        btnSeg.classList.toggle("active", mode === "seg");
        btnDet.classList.toggle("active", mode === "det");

        if (mode === "composite") {
            aiImage.src = currentArtifacts.composite_url;
            aiLayerTag.textContent = "AI Overlay (Boxes + U-Net++ Mask)";
        } else if (mode === "seg") {
            aiImage.src = currentArtifacts.seg_url;
            aiLayerTag.textContent = "U-Net++ Pleural Mask Only";
        } else if (mode === "det") {
            aiImage.src = currentArtifacts.det_url;
            aiLayerTag.textContent = "YOLO11m Detection Boxes Only";
        }
    }

    btnComposite.addEventListener("click", () => updateAILayerView("composite"));
    btnSeg.addEventListener("click", () => updateAILayerView("seg"));
    btnDet.addEventListener("click", () => updateAILayerView("det"));

    // 6. Drag & Drop Handlers
    dropzone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadAndPredict(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadAndPredict(e.dataTransfer.files[0]);
        }
    });

    function showLoading(show) {
        if (show) {
            loadingOverlay.classList.add("show");
        } else {
            loadingOverlay.classList.remove("show");
        }
    }

    // Initialize
    loadDemoCases();
});
