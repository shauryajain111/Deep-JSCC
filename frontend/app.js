/* ════════════════════════════════════════════════════════════
   Deep JSCC-Q Demo — Frontend JavaScript
   ════════════════════════════════════════════════════════════ */

const API = "http://localhost:5000/api";

/* ─── State ──────────────────────────────────────────────── */
let uploadedFile = null;
let sweepData    = null;
let psnrChart    = null;
let ssimChart    = null;

/* ─── DOM Refs ───────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const dropZone       = $("drop-zone");
const dropInner      = $("drop-inner");
const fileInput      = $("file-input");
const previewImg     = $("preview-img");
const snrSlider      = $("snr-slider");
const snrDisplay     = $("snr-display");
const jpegToggle     = $("jpeg-toggle");
const transmitBtn    = $("transmit-btn");
const sweepBtn       = $("sweep-btn");
const placeholder    = $("placeholder");
const spinnerWrap    = $("spinner-wrap");
const spinnerLabel   = $("spinner-label");
const imgGrid        = $("img-grid");
const metricBars     = $("metric-bars");
const chartSection   = $("chart-section");
const modelLabel     = $("model-label");
const modelInfo      = $("model-info");
const infoRows       = $("info-rows");

/* ════════════════════════════════════════════════════════════
   INITIALISATION
   ════════════════════════════════════════════════════════════ */
(async function init() {
  try {
    const res  = await fetch(`${API}/info`);
    const info = await res.json();

    modelLabel.textContent = `${info.model}  ·  k/n=${info.bandwidth_ratio}  ·  SNR_train=${info.trained_snr_db} dB`;

    const rows = [
      ["Model",          info.model],
      ["Image size",     info.img_size],
      ["Latent shape",   info.latent_shape],
      ["Bandwidth k/n",  info.bandwidth_ratio],
      ["Trained SNR",    `${info.trained_snr_db} dB`],
    ];
    infoRows.innerHTML = rows.map(([k, v]) =>
      `<div class="info-row"><span>${k}</span><span>${v}</span></div>`
    ).join("");
    modelInfo.style.display = "block";
  } catch {
    modelLabel.textContent = "⚠ Backend offline — start app.py";
  }
})();

/* ════════════════════════════════════════════════════════════
   SNR SLIDER
   ════════════════════════════════════════════════════════════ */
function updateSlider() {
  const min = +snrSlider.min, max = +snrSlider.max, val = +snrSlider.value;
  const pct = ((val - min) / (max - min) * 100).toFixed(1);
  snrSlider.style.setProperty("--pct", `${pct}%`);
  snrDisplay.textContent = `${val} dB`;
}
snrSlider.addEventListener("input", updateSlider);
updateSlider();

/* ════════════════════════════════════════════════════════════
   FILE UPLOAD — Drag & Drop + Click
   ════════════════════════════════════════════════════════════ */
function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showToast("⚠ Please upload a valid image file", "warn");
    return;
  }
  uploadedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.style.display = "block";
    dropInner.style.display  = "none";
  };
  reader.readAsDataURL(file);

  transmitBtn.disabled = false;
  sweepBtn.disabled    = false;

  // Reset results
  showPlaceholder();
}

fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

dropZone.addEventListener("click", e => {
  if (e.target === dropZone || e.target === dropInner || dropInner.contains(e.target)) {
    fileInput.click();
  }
});

["dragenter","dragover"].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("drag-over"); })
);
["dragleave","drop"].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("drag-over"); })
);
dropZone.addEventListener("drop", e => {
  handleFile(e.dataTransfer.files[0]);
});

/* ════════════════════════════════════════════════════════════
   TRANSMIT — single SNR
   ════════════════════════════════════════════════════════════ */
transmitBtn.addEventListener("click", async () => {
  if (!uploadedFile) return;

  showSpinner("Running Deep JSCC-Q inference…");

  const fd = new FormData();
  fd.append("file",       uploadedFile);
  fd.append("snr_db",     snrSlider.value);
  fd.append("show_jpeg",  jpegToggle.checked ? "true" : "false");

  try {
    const res  = await fetch(`${API}/transmit`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || res.statusText);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    hideSpinner();
    showPlaceholder();
    showToast(`❌ ${err.message}`, "error");
  }
});

/* ════════════════════════════════════════════════════════════
   SNR SWEEP
   ════════════════════════════════════════════════════════════ */
sweepBtn.addEventListener("click", async () => {
  if (!uploadedFile) return;

  showSpinner("Running SNR sweep (0 → 20 dB)…");

  const fd = new FormData();
  fd.append("file",       uploadedFile);
  fd.append("show_jpeg",  jpegToggle.checked ? "true" : "false");
  fd.append("snr_min",    "0");
  fd.append("snr_max",    "20");
  fd.append("snr_step",   "2");

  try {
    const res  = await fetch(`${API}/snr_sweep`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || res.statusText);
    }
    sweepData = await res.json();
    renderSweep(sweepData);
  } catch (err) {
    hideSpinner();
    showToast(`❌ ${err.message}`, "error");
  }
});

/* ════════════════════════════════════════════════════════════
   RENDER RESULTS
   ════════════════════════════════════════════════════════════ */
function renderResults(data) {
  hideSpinner();

  // Images
  $("img-orig").src  = `data:image/png;base64,${data.original_b64}`;
  $("img-jscc").src  = `data:image/png;base64,${data.jscc_b64}`;

  // JSCC metrics
  $("psnr-jscc").textContent = data.psnr_jscc.toFixed(2);
  $("ssim-jscc").textContent = data.ssim_jscc.toFixed(4);

  // JPEG
  const hasJpeg = !!data.jpeg_b64;
  $("card-jpeg").style.display = hasJpeg ? "" : "none";
  $("jpeg-legend-psnr").style.display  = hasJpeg ? "" : "none";
  $("jpeg-legend-psnr-text").style.display  = hasJpeg ? "" : "none";
  $("jpeg-legend-ssim").style.display  = hasJpeg ? "" : "none";
  $("jpeg-legend-ssim-text").style.display  = hasJpeg ? "" : "none";
  $("psnr-bar-jpeg").style.display     = hasJpeg ? "" : "none";
  $("ssim-bar-jpeg").style.display     = hasJpeg ? "" : "none";

  if (hasJpeg) {
    $("img-jpeg").src  = `data:image/png;base64,${data.jpeg_b64}`;
    $("psnr-jpeg").textContent = data.psnr_jpeg.toFixed(2);
    $("ssim-jpeg").textContent = data.ssim_jpeg.toFixed(4);
  }

  // Metric bars (PSNR clamped 0–45 dB, SSIM 0–1)
  animateBar("psnr-bar-jscc", data.psnr_jscc / 45 * 100);
  animateBar("ssim-bar-jscc", data.ssim_jscc * 100);
  if (hasJpeg) {
    animateBar("psnr-bar-jpeg", Math.max(0, data.psnr_jpeg) / 45 * 100);
    animateBar("ssim-bar-jpeg", Math.max(0, data.ssim_jpeg) * 100);
  }

  // Show sections
  placeholder.style.display   = "none";
  imgGrid.style.display        = "grid";
  metricBars.style.display     = "flex";
  chartSection.style.display   = "none"; // hide sweep if we just did single

  showToast(`✅ Transmitted at SNR = ${data.snr_db} dB`, "ok");
}

function animateBar(id, pct) {
  const el = $(id);
  el.style.width = "0%";
  requestAnimationFrame(() => {
    setTimeout(() => { el.style.width = `${Math.min(pct, 100).toFixed(1)}%`; }, 60);
  });
}

/* ════════════════════════════════════════════════════════════
   RENDER SWEEP CHARTS
   ════════════════════════════════════════════════════════════ */
function renderSweep(data) {
  hideSpinner();

  placeholder.style.display   = "none";
  imgGrid.style.display        = "none";
  metricBars.style.display     = "none";
  chartSection.style.display   = "block";

  const hasJpeg = data.psnr_jpeg && data.psnr_jpeg.length > 0;

  if (psnrChart) psnrChart.destroy();
  if (ssimChart) ssimChart.destroy();

  psnrChart = buildChart("psnr-chart", data.snr_values, [
    { label: "Deep JSCC-Q", data: data.psnr_jscc, color: "#48bb78" },
    ...(hasJpeg ? [{ label: "JPEG + BPSK", data: data.psnr_jpeg, color: "#f6ad55" }] : []),
  ], "PSNR (dB)", [0, 45]);

  ssimChart = buildChart("ssim-chart", data.snr_values, [
    { label: "Deep JSCC-Q", data: data.ssim_jscc, color: "#48bb78" },
    ...(hasJpeg ? [{ label: "JPEG + BPSK", data: data.ssim_jpeg, color: "#f6ad55" }] : []),
  ], "SSIM", [0, 1]);

  showToast("📈 SNR sweep complete!", "ok");
}

/* ─── Lightweight Canvas Chart ───────────────────────────── */
function buildChart(canvasId, xVals, series, yLabel, [yMin, yMax]) {
  const canvas = $(canvasId);
  const ctx    = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  const PAD  = { top: 20, right: 30, bottom: 45, left: 55 };
  const cW   = W - PAD.left - PAD.right;
  const cH   = H - PAD.top  - PAD.bottom;

  const xMin = Math.min(...xVals), xMax = Math.max(...xVals);

  function toX(v) { return PAD.left + (v - xMin) / (xMax - xMin) * cW; }
  function toY(v) { return PAD.top  + (1 - (v - yMin) / (yMax - yMin)) * cH; }

  // Clear
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "rgba(13,20,37,0.6)";
  ctx.beginPath();
  ctx.roundRect(0, 0, W, H, 10);
  ctx.fill();

  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = PAD.top + i / 5 * cH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cW, y); ctx.stroke();
  }
  for (let i = 0; i < xVals.length; i++) {
    const x = toX(xVals[i]);
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + cH); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top);
  ctx.lineTo(PAD.left, PAD.top + cH);
  ctx.lineTo(PAD.left + cW, PAD.top + cH);
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = "rgba(136,150,179,0.8)";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const v = yMin + (yMax - yMin) * (1 - i / 5);
    ctx.fillText(v.toFixed(yMax <= 1 ? 1 : 0), PAD.left - 8, PAD.top + i / 5 * cH + 4);
  }

  // X-axis labels
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(136,150,179,0.8)";
  xVals.forEach(v => {
    ctx.fillText(`${v}`, toX(v), PAD.top + cH + 18);
  });

  // Axis labels
  ctx.fillStyle = "rgba(136,150,179,0.9)";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Channel SNR (dB)", PAD.left + cW / 2, H - 5);

  ctx.save();
  ctx.translate(14, PAD.top + cH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  // Series
  series.forEach(({ label, data: vals, color }) => {
    // Area fill
    const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + cH);
    grad.addColorStop(0, color + "44");
    grad.addColorStop(1, color + "00");
    ctx.beginPath();
    ctx.moveTo(toX(xVals[0]), PAD.top + cH);
    xVals.forEach((x, i) => ctx.lineTo(toX(x), toY(vals[i])));
    ctx.lineTo(toX(xVals[xVals.length - 1]), PAD.top + cH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    xVals.forEach((x, i) => i === 0 ? ctx.moveTo(toX(x), toY(vals[i])) : ctx.lineTo(toX(x), toY(vals[i])));
    ctx.stroke();

    // Dots
    xVals.forEach((x, i) => {
      ctx.beginPath();
      ctx.arc(toX(x), toY(vals[i]), 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  });

  // Legend
  let lx = PAD.left + 10;
  series.forEach(({ label, color }) => {
    ctx.fillStyle = color;
    ctx.fillRect(lx, PAD.top + 4, 20, 4);
    ctx.fillStyle = "rgba(232,234,246,0.8)";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, lx + 26, PAD.top + 10);
    lx += ctx.measureText(label).width + 60;
  });

  // Return a minimal destroy handle
  return { destroy() { ctx.clearRect(0, 0, W, H); } };
}

/* ════════════════════════════════════════════════════════════
   UI HELPERS
   ════════════════════════════════════════════════════════════ */
function showPlaceholder() {
  placeholder.style.display  = "";
  imgGrid.style.display      = "none";
  metricBars.style.display   = "none";
  chartSection.style.display = "none";
  spinnerWrap.style.display  = "none";
}

function showSpinner(label = "Processing…") {
  placeholder.style.display  = "none";
  imgGrid.style.display      = "none";
  metricBars.style.display   = "none";
  chartSection.style.display = "none";
  spinnerWrap.style.display  = "flex";
  spinnerLabel.textContent   = label;
  transmitBtn.disabled = true;
  sweepBtn.disabled    = true;
}

function hideSpinner() {
  spinnerWrap.style.display = "none";
  transmitBtn.disabled = !uploadedFile;
  sweepBtn.disabled    = !uploadedFile;
}

/* ─── Toast Notification ─────────────────────────────────── */
let toastTimer = null;
function showToast(msg, type = "ok") {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
    const style = document.createElement("style");
    style.textContent = `
      .toast {
        position:fixed; bottom:2rem; left:50%; transform:translateX(-50%) translateY(100px);
        background:rgba(13,20,37,0.95); border:1px solid rgba(255,255,255,0.12);
        backdrop-filter:blur(20px);
        color:#e8eaf6; padding:0.7rem 1.4rem; border-radius:999px;
        font-size:0.85rem; font-weight:600; font-family:Inter,sans-serif;
        box-shadow:0 8px 32px rgba(0,0,0,0.5);
        transition:transform 0.35s cubic-bezier(0.4,0,0.2,1), opacity 0.35s;
        z-index:9999; opacity:0; white-space:nowrap;
      }
      .toast.show { transform:translateX(-50%) translateY(0); opacity:1; }
    `;
    document.head.appendChild(style);
  }
  toast.textContent = msg;
  toast.style.borderColor = type === "error" ? "#fc8181" : type === "warn" ? "#f6ad55" : "#48bb78";
  requestAnimationFrame(() => { toast.classList.add("show"); });
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.classList.remove("show"); }, 3500);
}
