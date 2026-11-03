/**
 * PneumoAI Diagnostic PACS Workstation 2.0 - Core Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Application State ---
    const state = {
        currentArtifacts: null,
        currentReport: null,
        currentLayer: "composite", // composite | seg | det
        currentLayout: "dual",     // dual | split | single
        currentPreset: "default",  // default | lung | invert | sharpen
        zoom: 1.0,
        pan: { x: 0, y: 0 },
        isPanning: false,
        panStart: { x: 0, y: 0 },
        splitPos: 50 // percentage
    };

    // --- DOM Elements ---
    const demoList = document.getElementById("demo-list");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const loadingOverlay = document.getElementById("loading-overlay");

    // Header & Meta Elements
    const metaExamId = document.getElementById("meta-exam-id");
    const metaDims = document.getElementById("meta-dims");
    const latencyChip = document.getElementById("latency-chip");
    const latencyText = document.getElementById("latency-text");

    // Viewport Containers
    const dualContainer = document.getElementById("dual-container");
    const splitContainer = document.getElementById("split-container");
    const rawImage = document.getElementById("raw-image");
    const aiImage = document.getElementById("ai-image");
    const splitRawImage = document.getElementById("split-raw-image");
    const splitAiImage = document.getElementById("split-ai-image");
    const splitOverlayClip = document.getElementById("split-overlay-clip");
    const splitDivider = document.getElementById("split-divider");
    const rawCanvasWrapper = document.getElementById("raw-canvas-wrapper");
    const aiCanvasWrapper = document.getElementById("ai-canvas-wrapper");

    // Tags & Badges in Viewport
    const rawInfoTag = document.getElementById("raw-info-tag");
    const aiLayerTag = document.getElementById("ai-layer-tag");
    const aiDetectionCount = document.getElementById("ai-detection-count");
    const zoomLevelText = document.getElementById("zoom-level-text");

    // Buttons & Controls
    const viewModeSelector = document.getElementById("view-mode-selector");
    const layerSelector = document.getElementById("layer-selector");
    const btnComposite = document.getElementById("btn-composite");
    const btnSeg = document.getElementById("btn-seg");
    const btnDet = document.getElementById("btn-det");

    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnZoomReset = document.getElementById("btn-zoom-reset");
    const btnWindowPreset = document.getElementById("btn-window-preset");
    const presetMenu = document.getElementById("preset-menu");
    const currentPresetLabel = document.getElementById("current-preset-label");

    // Report Elements
    const triageBanner = document.getElementById("triage-banner");
    const triageIcon = document.getElementById("triage-icon");
    const triageTitle = document.getElementById("triage-title");
    const triageUrgency = document.getElementById("triage-urgency");
    const triageSummary = document.getElementById("triage-summary");

    const ctrValue = document.getElementById("ctr-value");
    const ctrScaleFill = document.getElementById("ctr-scale-fill");
    const ctrStatus = document.getElementById("ctr-status");
    const collapseValue = document.getElementById("collapse-value");
    const collapseScaleFill = document.getElementById("collapse-scale-fill");
    const collapseStatus = document.getElementById("collapse-status");
    const collapseAction = document.getElementById("collapse-action");

    const findingsTableBody = document.getElementById("findings-tbody");
    const lesionsCount = document.getElementById("lesions-count");
    const impressionText = document.getElementById("impression-text");

    // Header Tools
    const btnFullscreen = document.getElementById("btn-fullscreen");
    const btnShortcuts = document.getElementById("btn-shortcuts");
    const shortcutsModal = document.getElementById("shortcuts-modal");
    const modalCloseBtn = document.getElementById("modal-close-btn");
    const btnCopyReport = document.getElementById("btn-copy-report");
    const btnPrintReport = document.getElementById("btn-print-report");
    const toastContainer = document.getElementById("toast-container");

    // =========================================================================
    // 1. Toast Notification Utility
    // =========================================================================
    function showToast(message) {
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 2800);
    }

    // =========================================================================
    // 2. Fetch & Render Demo Benchmark Cases
    // =========================================================================
    async function loadDemoCases() {
        try {
            const res = await fetch("/api/demo-cases");
            const data = await res.json();
            renderDemoCases(data.cases || []);

            // Automatically run the first benchmark case
            if (data.cases && data.cases.length > 0) {
                runDemoCase(data.cases[0].filename);
            }
        } catch (err) {
            console.error("Failed to fetch demo cases:", err);
            showToast("Failed to load demo cases catalog.");
        }
    }

    function renderDemoCases(cases) {
        demoList.innerHTML = "";
        cases.forEach((c, idx) => {
            const card = document.createElement("div");
            card.className = `demo-card ${idx === 0 ? "active" : ""}`;
            card.dataset.filename = c.filename;

            let badgeHtml = "";
            if (c.badge) {
                badgeHtml = `<span class="demo-badge ${c.badge}">${c.badge}</span>`;
            }

            card.innerHTML = `
                <img src="${c.thumbnail_url}" class="demo-thumb" alt="${c.title}" />
                <div class="demo-info">
                    <div class="demo-title-row">
                        <div class="demo-title">${c.title}</div>
                        ${badgeHtml}
                    </div>
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

    // =========================================================================
    // 3. Execution & Inference Handlers
    // =========================================================================
    async function runDemoCase(filename) {
        showLoading(true);
        try {
            const res = await fetch("/api/predict/demo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename })
            });
            if (!res.ok) throw new Error(await res.text());
            const report = await res.json();
            handleReportData(report);
            showToast(`Loaded: ${report.exam_id}`);
        } catch (err) {
            console.error("Inference failed:", err);
            showToast(`Error: ${err.message}`);
        } finally {
            showLoading(false);
        }
    }

    async function uploadAndPredict(file) {
        showLoading(true);
        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/predict/upload", {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error(await res.text());
            const report = await res.json();

            // Clear active demo selection
            document.querySelectorAll(".demo-card").forEach(el => el.classList.remove("active"));
            handleReportData(report);
            showToast(`Uploaded: ${file.name}`);
        } catch (err) {
            console.error("Upload inference failed:", err);
            showToast(`Upload error: ${err.message}`);
        } finally {
            showLoading(false);
        }
    }

    // =========================================================================
    // 4. Update UI with Diagnostic Report
    // =========================================================================
    function handleReportData(report) {
        state.currentReport = report;
        state.currentArtifacts = report.artifacts;

        // Reset Pan & Zoom on new case
        resetZoomPan();

        // 1. Header Metadata
        metaExamId.textContent = report.exam_id;
        metaDims.textContent = report.dimensions;
        rawInfoTag.textContent = report.source_image;

        // Latency
        const lat = report.latency;
        latencyText.textContent = `${lat.total_ms}ms (U-Net: ${lat.unet_ms}ms | YOLO: ${lat.yolo_ms}ms)`;

        // 2. Viewport Images
        rawImage.src = report.artifacts.raw_url;
        splitRawImage.src = report.artifacts.raw_url;
        updateAILayerView(state.currentLayer);

        // 3. Triage Alert Card
        const alertInfo = report.triage_alert;
        triageBanner.className = `triage-card ${alertInfo.badge_class}`;
        triageTitle.textContent = alertInfo.level;
        triageSummary.textContent = alertInfo.summary;

        if (alertInfo.level === "EMERGENCY") {
            triageIcon.textContent = "🚨";
            triageUrgency.textContent = "IMMEDIATE ATTENTION";
        } else if (alertInfo.level === "PRIORITY") {
            triageIcon.textContent = "⚠️";
            triageUrgency.textContent = "ELEVATED PRIORITY";
        } else {
            triageIcon.textContent = "✓";
            triageUrgency.textContent = "ROUTINE CLINICAL QUEUE";
        }

        // 4. Cardiothoracic Ratio (CTR)
        const ctr = report.biomarkers.cardiothoracic_ratio;
        if (ctr && ctr.ctr_value !== null) {
            ctrValue.textContent = ctr.ctr_value.toFixed(2);
            ctrStatus.textContent = ctr.status;
            // Map 0.3 -> 0%, 0.7 -> 100%
            const pct = Math.min(100, Math.max(0, ((ctr.ctr_value - 0.3) / 0.4) * 100));
            ctrScaleFill.style.width = `${pct}%`;
            ctrScaleFill.className = ctr.is_enlarged ? "scale-fill fill-amber" : "scale-fill";
        } else {
            ctrValue.textContent = "---";
            ctrStatus.textContent = "Normal / Within Limits";
            ctrScaleFill.style.width = "40%";
            ctrScaleFill.className = "scale-fill";
        }

        // 5. Pleural Collapse (Phi)
        const ptx = report.biomarkers.pleural_collapse;
        collapseValue.textContent = `${ptx.collapse_percentage}%`;
        collapseStatus.textContent = ptx.severity;
        const collapsePct = Math.min(100, (ptx.collapse_percentage / 30.0) * 100);
        collapseScaleFill.style.width = `${collapsePct}%`;

        if (ptx.detected && ptx.intervention_recommended) {
            collapseAction.style.display = "block";
            collapseAction.textContent = `Action: ${ptx.intervention_recommended}`;
        } else {
            collapseAction.style.display = "none";
        }

        // 6. Findings Table
        findingsTableBody.innerHTML = "";
        const localized = report.findings.localized_lesions;
        let totalCount = localized.length + (ptx.detected ? 1 : 0);
        lesionsCount.textContent = `${totalCount} item${totalCount !== 1 ? "s" : ""}`;
        aiDetectionCount.textContent = `${totalCount} Lesions Delineated`;

        // If Pneumothorax detected, add as first row
        if (ptx.detected) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><span class="path-badge" style="background: rgba(0, 255, 127, 0.18); color: #00ff7f; border: 1px solid rgba(0, 255, 127, 0.4);">Pneumothorax</span></td>
                <td><strong style="color: #00ff7f;">${(ptx.confidence_mean * 100).toFixed(1)}%</strong></td>
                <td>Visceral Pleura</td>
                <td>${ptx.intervention_recommended}</td>
            `;
            findingsTableBody.appendChild(tr);
        }

        // Add Localized YOLO Lesions
        localized.forEach(det => {
            const tr = document.createElement("tr");
            let badgeBg = "rgba(56, 189, 248, 0.15)";
            let badgeColor = "#38bdf8";
            let borderColor = "rgba(56, 189, 248, 0.3)";

            if (det.class_name.includes("Cardiomegaly")) {
                badgeBg = "rgba(239, 68, 68, 0.15)";
                badgeColor = "#ef4444";
                borderColor = "rgba(239, 68, 68, 0.3)";
            } else if (det.class_name.includes("Aortic")) {
                badgeBg = "rgba(249, 115, 22, 0.15)";
                badgeColor = "#f97316";
                borderColor = "rgba(249, 115, 22, 0.3)";
            } else if (det.class_name.includes("Parenchymal")) {
                badgeBg = "rgba(168, 85, 247, 0.15)";
                badgeColor = "#c084fc";
                borderColor = "rgba(168, 85, 247, 0.3)";
            }

            tr.innerHTML = `
                <td><span class="path-badge" style="background: ${badgeBg}; color: ${badgeColor}; border: 1px solid ${borderColor};">${det.class_name}</span></td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td>${det.lung_zone}</td>
                <td>Radiological Correlation</td>
            `;
            findingsTableBody.appendChild(tr);
        });

        if (totalCount === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--text-dim); padding: 1.25rem;">No acute focal lesions or pleural detachment detected.</td>`;
            findingsTableBody.appendChild(tr);
        }

        // 7. Generate RadLex Structured Text
        generateImpression(report);
    }

    function generateImpression(report) {
        const ptx = report.findings.pneumothorax_segmentation;
        const ctr = report.biomarkers.cardiothoracic_ratio;
        const lesions = report.findings.localized_lesions;

        let lines = [];
        lines.push(`EXAMINATION: CHEST RADIOGRAPH (PA VIEW)`);
        lines.push(`ACCESSION / EXAM ID: ${report.exam_id}`);
        lines.push(`TRIAGE PRIORITY: [${report.triage_alert.level}]`);
        lines.push(`----------------------------------------------------------------`);
        lines.push(`FINDINGS:`);

        if (ptx.detected) {
            lines.push(`• PLEURA & THORACIC CAGE: Pneumothorax confirmed by visceral pleural line delineation. Ipsilateral lung collapse quantified at ${ptx.collapse_percentage}% (${ptx.severity}). Recommended: ${ptx.intervention_recommended}.`);
        } else {
            lines.push(`• PLEURA & THORACIC CAGE: No pneumothorax, effusion, or apical pleural line detachment.`);
        }

        if (ctr && ctr.ctr_value) {
            lines.push(`• CARDIOMEDIASTINAL SILHOUETTE: CTR = ${ctr.ctr_value.toFixed(2)} (${ctr.status}).`);
        } else {
            lines.push(`• CARDIOMEDIASTINAL SILHOUETTE: Cardiothoracic ratio within normal physiological limits.`);
        }

        if (lesions.length > 0) {
            const nonCardiac = lesions.filter(l => !l.class_name.includes("Cardiomegaly"));
            if (nonCardiac.length > 0) {
                const listStr = nonCardiac.map(l => `${l.class_name} localized in ${l.lung_zone} (conf: ${l.confidence})`).join("; ");
                lines.push(`• PULMONARY PARENCHYMA: ${listStr}.`);
            }
        }

        lines.push(`\nIMPRESSION: Automated consensus synthesis formatted in accordance with RadLex standard terminology.`);
        impressionText.textContent = lines.join("\n");
    }

    // =========================================================================
    // 5. Layer Switching (Composite / Pleura Only / Detection Only)
    // =========================================================================
    function updateAILayerView(layer) {
        if (!state.currentArtifacts) return;
        state.currentLayer = layer;

        btnComposite.classList.toggle("active", layer === "composite");
        btnSeg.classList.toggle("active", layer === "seg");
        btnDet.classList.toggle("active", layer === "det");

        let activeUrl = state.currentArtifacts.composite_url;
        let layerLabel = "AI Overlay (Boxes + U-Net++ Mask)";

        if (layer === "seg") {
            activeUrl = state.currentArtifacts.seg_url;
            layerLabel = "U-Net++ Pleural Boundary Mask Only";
        } else if (layer === "det") {
            activeUrl = state.currentArtifacts.det_url;
            layerLabel = "YOLO11m Detection Boxes Only";
        }

        aiImage.src = activeUrl;
        splitAiImage.src = activeUrl;
        aiLayerTag.textContent = layerLabel;
    }

    btnComposite.addEventListener("click", () => updateAILayerView("composite"));
    btnSeg.addEventListener("click", () => updateAILayerView("seg"));
    btnDet.addEventListener("click", () => updateAILayerView("det"));

    // =========================================================================
    // 6. Layout Mode Switching (Dual View / Curtain Split / Single Focus)
    // =========================================================================
    viewModeSelector.querySelectorAll(".segmented-item").forEach(btn => {
        btn.addEventListener("click", () => {
            const viewMode = btn.dataset.view;
            setLayoutMode(viewMode);
        });
    });

    function setLayoutMode(mode) {
        state.currentLayout = mode;
        viewModeSelector.querySelectorAll(".segmented-item").forEach(b => {
            b.classList.toggle("active", b.dataset.view === mode);
        });

        if (mode === "dual") {
            dualContainer.style.display = "grid";
            dualContainer.classList.remove("single-mode");
            splitContainer.style.display = "none";
        } else if (mode === "single") {
            dualContainer.style.display = "grid";
            dualContainer.classList.add("single-mode");
            splitContainer.style.display = "none";
        } else if (mode === "split") {
            dualContainer.style.display = "none";
            splitContainer.style.display = "block";
            updateSplitPosition(state.splitPos);
        }
    }

    // =========================================================================
    // 7. Interactive Curtain Split Slider
    // =========================================================================
    let isDraggingSplit = false;

    splitDivider.addEventListener("mousedown", (e) => {
        isDraggingSplit = true;
        e.preventDefault();
    });

    window.addEventListener("mouseup", () => {
        isDraggingSplit = false;
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDraggingSplit || state.currentLayout !== "split") return;
        const rect = splitContainer.querySelector(".split-image-wrapper").getBoundingClientRect();
        let x = e.clientX - rect.left;
        let pct = (x / rect.width) * 100;
        pct = Math.max(5, Math.min(95, pct));
        updateSplitPosition(pct);
    });

    function updateSplitPosition(pct) {
        state.splitPos = pct;
        splitOverlayClip.style.width = `${pct}%`;
        splitDivider.style.left = `${pct}%`;
    }

    // =========================================================================
    // 8. Pan & Zoom Canvas Controls
    // =========================================================================
    function applyTransform() {
        const transformStr = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
        rawImage.style.transform = transformStr;
        aiImage.style.transform = transformStr;
        zoomLevelText.textContent = `${Math.round(state.zoom * 100)}%`;
    }

    function resetZoomPan() {
        state.zoom = 1.0;
        state.pan = { x: 0, y: 0 };
        applyTransform();
    }

    btnZoomIn.addEventListener("click", () => {
        state.zoom = Math.min(3.5, state.zoom + 0.25);
        applyTransform();
    });

    btnZoomOut.addEventListener("click", () => {
        state.zoom = Math.max(0.75, state.zoom - 0.25);
        applyTransform();
    });

    btnZoomReset.addEventListener("click", resetZoomPan);

    // Mouse Wheel Zoom
    [rawCanvasWrapper, aiCanvasWrapper].forEach(wrapper => {
        wrapper.addEventListener("wheel", (e) => {
            e.preventDefault();
            const delta = e.deltaY < 0 ? 0.15 : -0.15;
            state.zoom = Math.min(3.5, Math.max(0.75, state.zoom + delta));
            applyTransform();
        }, { passive: false });

        // Drag to Pan
        wrapper.addEventListener("mousedown", (e) => {
            state.isPanning = true;
            state.panStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
            wrapper.classList.add("grabbing");
        });
    });

    window.addEventListener("mousemove", (e) => {
        if (!state.isPanning) return;
        state.pan = {
            x: e.clientX - state.panStart.x,
            y: e.clientY - state.panStart.y
        };
        applyTransform();
    });

    window.addEventListener("mouseup", () => {
        if (state.isPanning) {
            state.isPanning = false;
            rawCanvasWrapper.classList.remove("grabbing");
            aiCanvasWrapper.classList.remove("grabbing");
        }
    });

    // =========================================================================
    // 9. Window / Level Contrast Presets
    // =========================================================================
    btnWindowPreset.addEventListener("click", (e) => {
        e.stopPropagation();
        presetMenu.classList.toggle("show");
    });

    document.addEventListener("click", () => {
        presetMenu.classList.remove("show");
    });

    presetMenu.querySelectorAll(".preset-opt").forEach(opt => {
        opt.addEventListener("click", () => {
            const preset = opt.dataset.preset;
            applyPreset(preset, opt.textContent);
        });
    });

    function applyPreset(preset, label) {
        state.currentPreset = preset;
        currentPresetLabel.textContent = label;

        presetMenu.querySelectorAll(".preset-opt").forEach(o => {
            o.classList.toggle("active", o.dataset.preset === preset);
        });

        // Remove old filter classes
        [rawImage, aiImage, splitRawImage, splitAiImage].forEach(img => {
            img.classList.remove("filter-lung", "filter-invert", "filter-sharpen");
            if (preset !== "default") {
                img.classList.add(`filter-${preset}`);
            }
        });

        showToast(`Preset: ${label}`);
    }

    // =========================================================================
    // 10. File Upload & Drag-and-Drop
    // =========================================================================
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

    // =========================================================================
    // 11. Copy Report & Print Tools
    // =========================================================================
    btnCopyReport.addEventListener("click", () => {
        const text = impressionText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            showToast("RadLex Report copied to clipboard!");
        }).catch(err => {
            showToast("Failed to copy report.");
        });
    });

    btnPrintReport.addEventListener("click", () => {
        window.print();
    });

    // Fullscreen Toggle
    btnFullscreen.addEventListener("click", () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                showToast("Fullscreen unavailable.");
            });
        } else {
            document.exitFullscreen();
        }
    });

    // Shortcuts Modal
    btnShortcuts.addEventListener("click", () => {
        shortcutsModal.style.display = "flex";
    });

    modalCloseBtn.addEventListener("click", () => {
        shortcutsModal.style.display = "none";
    });

    shortcutsModal.addEventListener("click", (e) => {
        if (e.target === shortcutsModal) {
            shortcutsModal.style.display = "none";
        }
    });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        if (e.key === "1") updateAILayerView("composite");
        else if (e.key === "2") updateAILayerView("seg");
        else if (e.key === "3") updateAILayerView("det");
        else if (e.key.toLowerCase() === "d") setLayoutMode("dual");
        else if (e.key.toLowerCase() === "s") setLayoutMode("split");
        else if (e.key.toLowerCase() === "r") resetZoomPan();
        else if (e.key.toLowerCase() === "i") {
            const nextPreset = state.currentPreset === "invert" ? "default" : "invert";
            applyPreset(nextPreset, nextPreset === "invert" ? "Invert Film" : "Standard Thoracic");
        } else if (e.key === "+" || e.key === "=") {
            state.zoom = Math.min(3.5, state.zoom + 0.25);
            applyTransform();
        } else if (e.key === "-") {
            state.zoom = Math.max(0.75, state.zoom - 0.25);
            applyTransform();
        } else if (e.key === "Escape") {
            shortcutsModal.style.display = "none";
        }
    });

    function showLoading(show) {
        if (show) {
            loadingOverlay.classList.add("show");
        } else {
            loadingOverlay.classList.remove("show");
        }
    }

    // --- Startup Initialization ---
    loadDemoCases();
});
