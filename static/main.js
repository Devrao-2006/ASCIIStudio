/* ═══════════════════════════════════════════════════════
   ASCII STUDIO — Frontend Logic
   WebSocket-driven parameter updates + Image/Video Convert + Studio Tabs
   ═══════════════════════════════════════════════════════ */

const WS_URL = `ws://${location.host}/ws`;

// ── WebSocket ────────────────────────────────────────────
let ws = null;
let wsReady = false;
const pendingUpdates = {};

function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsReady = true;
    console.log("WS connected");
  };

  ws.onclose = () => {
    wsReady = false;
    setTimeout(connectWS, 1000);
  };

  ws.onerror = (e) => console.warn("WS error:", e);
}

function sendParam(key, value) {
  pendingUpdates[key] = value;
  if (currentTab === "image" && currentImageFile) {
    scheduleImageRerender();
  }
}

function flushUpdates() {
  if (wsReady && Object.keys(pendingUpdates).length > 0) {
    ws.send(JSON.stringify({ ...pendingUpdates }));
    for (const k in pendingUpdates) delete pendingUpdates[k];
  }
  requestAnimationFrame(flushUpdates);
}

// ── Studio Tabs Navigation ───────────────────────────────
let currentTab = "webcam";
const statusPill = document.getElementById("status-pill");

function initStudioTabs() {
  document.querySelectorAll(".studio-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".studio-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".studio-view").forEach(v => v.classList.remove("active"));

      tab.classList.add("active");
      currentTab = tab.dataset.tab;
      const targetView = document.getElementById(`view-${currentTab}`);
      if (targetView) targetView.classList.add("active");

      const feed = document.getElementById("feed");
      if (currentTab === "webcam") {
        statusPill.textContent = "● LIVE";
        statusPill.style.color = "var(--green)";
        if (feed) feed.src = "/video_feed";
        fetch("/api/camera/resume", { method: "POST" }).catch(console.warn);
      } else {
        if (feed) feed.src = "";
        fetch("/api/camera/pause", { method: "POST" }).catch(console.warn);
        if (currentTab === "image") {
          statusPill.textContent = "🖼️ PHOTO STUDIO";
          statusPill.style.color = "var(--accent)";
        } else if (currentTab === "video") {
          statusPill.textContent = "🎬 VIDEO STUDIO";
          statusPill.style.color = "var(--accent)";
        }
      }
    });
  });
}

// ── Slider controls ──────────────────────────────────────
function initSliders() {
  document.querySelectorAll(".control").forEach(ctrl => {
    const param   = ctrl.dataset.param;
    const realMin = parseFloat(ctrl.dataset.min);
    const realMax = parseFloat(ctrl.dataset.max);
    const realStep= parseFloat(ctrl.dataset.step);
    const isInt   = ctrl.dataset.int === "true";
    const valEl   = ctrl.querySelector(".val");
    const input   = ctrl.querySelector("input[type=range]");

    const steps = Math.round((realMax - realMin) / realStep);
    input.min  = 0;
    input.max  = steps;
    input.step = 1;

    const defaultReal = parseFloat(ctrl.dataset.default);
    input.value = Math.round((defaultReal - realMin) / realStep);

    function updateFromSlider() {
      const ticks = parseInt(input.value);
      let real = realMin + ticks * realStep;
      if (param === "zoom") real = Math.max(1.0, real);
      real = Math.round(real * 1000) / 1000;

      const display = isInt ? String(Math.round(real))
                            : real.toFixed(realStep < 0.01 ? 3 : realStep < 0.1 ? 2 : 1);
      valEl.textContent = param === "zoom" ? `${display}×` : display;
      sendParam(param, isInt ? Math.round(real) : real);
    }

    input.addEventListener("input", updateFromSlider);
    updateFromSlider();
  });
}

// ── Toggle buttons ───────────────────────────────────────
function initToggles() {
  document.querySelectorAll(".toggle-btn").forEach(btn => {
    const isOn = btn.dataset.val === "true";
    btn.classList.toggle("active", isOn);
    btn.textContent = isOn ? "ON" : "OFF";

    btn.addEventListener("click", () => {
      const current = btn.dataset.val === "true";
      const next    = !current;
      btn.dataset.val = String(next);
      btn.classList.toggle("active", next);
      btn.textContent = next ? "ON" : "OFF";
      sendParam(btn.dataset.param, next);
    });
  });
}

// ── Theme selector ───────────────────────────────────────
let currentThemeName = "Classic";
const THEME_NAMES = ["Classic", "Amber", "Gold", "Ghost", "Warm"];

function initThemes() {
  document.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const themeIdx = parseInt(btn.dataset.theme);
      currentThemeName = THEME_NAMES[themeIdx] || "Theme";
      sendParam("theme", themeIdx);
    });
  });
}

// ── View mode buttons ────────────────────────────────────
function initViewModes() {
  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = parseInt(btn.dataset.mode);
      sendParam("view_mode", mode);
      sendParam("view_uncompressed", mode === 3);
    });
  });
}

// ── Video Recording Logic (Live Webcam) ───────────────────
const btnRecord = document.getElementById("btn-record");
const recBadge  = document.getElementById("rec-badge");
const recTimer  = document.getElementById("rec-timer");
let isRecording = false;
let recordInterval = null;
let recordStartTime = 0;

async function toggleRecording() {
  if (!isRecording) {
    try {
      const res = await fetch("/api/record/start", { method: "POST" });
      const data = await res.json();
      isRecording = true;
      recordStartTime = Date.now();
      btnRecord.classList.add("recording");
      btnRecord.setAttribute("data-tooltip", "Stop Recording (R)");
      recBadge.classList.add("active");
      updateRecTimer();
      recordInterval = setInterval(updateRecTimer, 1000);
      console.log("Recording started:", data.filename);
    } catch (e) {
      console.error("Failed to start recording:", e);
    }
  } else {
    try {
      const res = await fetch("/api/record/stop", { method: "POST" });
      const data = await res.json();
      isRecording = false;
      clearInterval(recordInterval);
      btnRecord.classList.remove("recording");
      btnRecord.setAttribute("data-tooltip", "Record ASCII Video (R)");
      recBadge.classList.remove("active");
      alert(`Video saved: app/captures/videos/${data.filename}`);
    } catch (e) {
      console.error("Failed to stop recording:", e);
    }
  }
}

function updateRecTimer() {
  const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
  const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
  const s = String(elapsedSec % 60).padStart(2, '0');
  recTimer.textContent = `${m}:${s}`;
}

if (btnRecord) btnRecord.addEventListener("click", toggleRecording);

// ── Image Convert & Export Studio ─────────────────────────
let currentImageFile = null;
let imageRerenderTimer = null;
let currentRenderedImgBlob = null;

const imgDropzone    = document.getElementById("img-dropzone");
const imgFileInput   = document.getElementById("img-file-input");
const btnBrowseImg   = document.getElementById("btn-browse-img");
const imgPreviewWrap = document.getElementById("img-preview-wrap");
const imgPreviewEl   = document.getElementById("img-rendered-preview");
const btnChangeImg   = document.getElementById("btn-change-img");
const btnDownloadImg = document.getElementById("btn-download-img");

function initImageStudio() {
  btnBrowseImg.addEventListener("click", () => imgFileInput.click());
  imgDropzone.addEventListener("click", (e) => {
    if (e.target !== btnBrowseImg) imgFileInput.click();
  });

  imgFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
    }
  });

  // Drag & drop handlers
  ["dragenter", "dragover"].forEach(eventName => {
    imgDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      imgDropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(eventName => {
    imgDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      imgDropzone.classList.remove("dragover");
    });
  });
  imgDropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  });

  btnChangeImg.addEventListener("click", () => {
    currentImageFile = null;
    currentRenderedImgBlob = null;
    imgPreviewWrap.style.display = "none";
    imgDropzone.style.display = "flex";
    imgFileInput.value = "";
  });

  btnDownloadImg.addEventListener("click", () => {
    if (!currentRenderedImgBlob) return;
    const url = URL.createObjectURL(currentRenderedImgBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `asciiphoto_${currentThemeName.toLowerCase()}_${Date.now()}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  });
}

function handleImageFile(file) {
  currentImageFile = file;
  renderActiveImage();
}

function scheduleImageRerender() {
  clearTimeout(imageRerenderTimer);
  imageRerenderTimer = setTimeout(renderActiveImage, 150);
}

async function renderActiveImage() {
  if (!currentImageFile) return;
  try {
    const formData = new FormData();
    formData.append("file", currentImageFile);

    const res = await fetch("/api/convert/image", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Image processing failed");
    const blob = await res.blob();
    currentRenderedImgBlob = blob;

    const previewUrl = URL.createObjectURL(blob);
    imgPreviewEl.src = previewUrl;
    imgDropzone.style.display = "none";
    imgPreviewWrap.style.display = "flex";
  } catch (err) {
    console.error("Image render error:", err);
  }
}

// ── Video Convert & Export Studio ─────────────────────────
let currentVideoFile = null;
let videoPollInterval = null;

const vidDropzone         = document.getElementById("vid-dropzone");
const vidFileInput        = document.getElementById("vid-file-input");
const btnBrowseVid        = document.getElementById("btn-browse-vid");
const vidJobPanel         = document.getElementById("vid-job-panel");
const vidJobName          = document.getElementById("vid-job-name");
const vidJobStatus        = document.getElementById("vid-job-status");
const btnChangeVid        = document.getElementById("btn-change-vid");
const btnStartVidConvert  = document.getElementById("btn-start-vid-convert");
const vidProgressWrap     = document.getElementById("vid-progress-wrap");
const vidProgressFill     = document.getElementById("vid-progress-fill");
const vidProgressPct      = document.getElementById("vid-progress-pct");
const vidFramesInfo       = document.getElementById("vid-frames-info");
const btnDownloadVid      = document.getElementById("btn-download-vid");

function initVideoStudio() {
  btnBrowseVid.addEventListener("click", () => vidFileInput.click());
  vidDropzone.addEventListener("click", (e) => {
    if (e.target !== btnBrowseVid) vidFileInput.click();
  });

  vidFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleVideoFile(e.target.files[0]);
    }
  });

  // Drag & drop handlers
  ["dragenter", "dragover"].forEach(eventName => {
    vidDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      vidDropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(eventName => {
    vidDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      vidDropzone.classList.remove("dragover");
    });
  });
  vidDropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleVideoFile(e.dataTransfer.files[0]);
    }
  });

  btnChangeVid.addEventListener("click", () => {
    clearInterval(videoPollInterval);
    currentVideoFile = null;
    vidJobPanel.style.display = "none";
    vidDropzone.style.display = "flex";
    vidFileInput.value = "";
    btnStartVidConvert.style.display = "inline-flex";
    btnStartVidConvert.disabled = false;
    btnDownloadVid.style.display = "none";
    vidProgressWrap.style.display = "none";
  });

  btnStartVidConvert.addEventListener("click", startVideoConversion);
}

function handleVideoFile(file) {
  currentVideoFile = file;
  vidJobName.textContent = file.name;
  vidJobStatus.textContent = `Size: ${(file.size / (1024*1024)).toFixed(1)} MB • Theme: ${currentThemeName}`;
  vidDropzone.style.display = "none";
  vidJobPanel.style.display = "flex";
  vidProgressWrap.style.display = "none";
  btnStartVidConvert.style.display = "inline-flex";
  btnStartVidConvert.disabled = false;
  btnDownloadVid.style.display = "none";
}

async function startVideoConversion() {
  if (!currentVideoFile) return;

  btnStartVidConvert.disabled = true;
  vidProgressWrap.style.display = "flex";
  vidProgressFill.style.width = "0%";
  vidProgressPct.textContent = "0%";
  vidFramesInfo.textContent = "Starting GPU render...";
  vidJobStatus.textContent = `Rendering frame-by-frame (${currentThemeName} Theme)...`;

  try {
    const formData = new FormData();
    formData.append("file", currentVideoFile);

    const res = await fetch("/api/convert/video", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Failed to start video conversion");
    const data = await res.json();
    const jobId = data.job_id;

    // Poll status
    videoPollInterval = setInterval(async () => {
      try {
        const sRes = await fetch(`/api/convert/video/status/${jobId}`);
        const job = await sRes.json();

        if (job.status === "processing") {
          vidProgressFill.style.width = `${job.progress}%`;
          vidProgressPct.textContent = `${job.progress}%`;
          vidFramesInfo.textContent = `Frame ${job.current_frame} / ${job.total_frames || '?'}`;
        } else if (job.status === "completed") {
          clearInterval(videoPollInterval);
          vidProgressFill.style.width = "100%";
          vidProgressPct.textContent = "100%";
          vidFramesInfo.textContent = "Complete!";
          vidJobStatus.textContent = `Rendering complete • Saved as ${job.filename}`;
          btnStartVidConvert.style.display = "none";
          btnDownloadVid.style.display = "inline-flex";
          btnDownloadVid.onclick = () => {
            window.location.href = `/api/convert/video/download/${jobId}`;
          };
        } else if (job.status === "error") {
          clearInterval(videoPollInterval);
          vidJobStatus.textContent = `Error: ${job.error || 'Conversion failed'}`;
          btnStartVidConvert.disabled = false;
        }
      } catch (pollErr) {
        console.warn("Poll error:", pollErr);
      }
    }, 500);

  } catch (err) {
    console.error("Video conversion error:", err);
    vidJobStatus.textContent = "Error starting conversion.";
    btnStartVidConvert.disabled = false;
  }
}

// ── Fullscreen & Keyboard Shortcuts ──────────────────────
const videoWrap = document.getElementById("video-wrap");
const btnFS     = document.getElementById("btn-fullscreen");

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(console.warn);
  } else {
    document.exitFullscreen().catch(console.warn);
  }
}

if (btnFS) btnFS.addEventListener("click", toggleFullscreen);
if (videoWrap) videoWrap.addEventListener("dblclick", toggleFullscreen);

function takeSnapshot() {
  const feed = document.getElementById("feed");
  const canvas = document.createElement("canvas");
  canvas.width  = feed.naturalWidth  || feed.width;
  canvas.height = feed.naturalHeight || feed.height;
  canvas.getContext("2d").drawImage(feed, 0, 0);
  const a = document.createElement("a");
  a.href     = canvas.toDataURL("image/png");
  a.download = `asciicam_${currentThemeName.toLowerCase()}_${Date.now()}.png`;
  a.click();
}

const btnSnapshot = document.getElementById("btn-snapshot");
if (btnSnapshot) btnSnapshot.addEventListener("click", takeSnapshot);

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "f" || e.key === "F") toggleFullscreen();
  if (e.key === "r" || e.key === "R") toggleRecording();
  if (e.key === "s" || e.key === "S") takeSnapshot();
  if (e.key === "Escape" && document.fullscreenElement) {
    document.exitFullscreen().catch(console.warn);
  }
});

// ── Boot ─────────────────────────────────────────────────
connectWS();
initStudioTabs();
initSliders();
initToggles();
initThemes();
initViewModes();
initImageStudio();
initVideoStudio();
requestAnimationFrame(flushUpdates);
