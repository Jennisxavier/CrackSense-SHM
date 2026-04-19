"""
==============================================================================
  CRACK DETECTION & STRUCTURAL HEALTH MONITORING — STREAMLIT DASHBOARD
  Wraps: ResNet50 → YOLOv8 → 501-pt Profile → DLM → GRU RUL Engine
==============================================================================
"""

import tensorflow as tf
import os
import io
import math
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import streamlit as st

# ── Reliable OpenCV loader (handles TIF/BMP path issues on Windows) ───────────
def cv2_load(path, flag=cv2.IMREAD_COLOR):
    """Falls back to imdecode-from-bytes if imread returns None (e.g. TIF on Windows)."""
    img = cv2.imread(path, flag)
    if img is None:
        try:
            buf = np.frombuffer(open(path, 'rb').read(), dtype=np.uint8)
            img = cv2.imdecode(buf, flag)
        except Exception:
            pass
    return img

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # d:\CRACKAPP

CONFIG = {
    "resnet_model" : os.path.join(BASE_DIR, "resnet_crack_classifier (1).keras"),
    "yolo_model"   : os.path.join(BASE_DIR, "best (1).pt"),
    "dlm_model"    : os.path.join(BASE_DIR, "krknet_replica (3).keras"),
    "gru_model"    : os.path.join(BASE_DIR, "gru_growth_model (1).keras"),
    "csv_database" : os.path.join(BASE_DIR, "krkCMd_table_with_growth - krkCMd_table_with_growth.csv"),
    "scaler_X"     : os.path.join(BASE_DIR, "scaler_X.pkl"),
    "scaler_y"     : os.path.join(BASE_DIR, "scaler_y.pkl"),
    "pi2mi"        : 1000 * 25.4 / 6400,
}

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CrackSense — SHM Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PWA Injection ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="manifest" href="/static/manifest.json">
<link rel="apple-touch-icon" href="/static/icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/static/sw.js').then(function(registration) {
        console.log('PWA ServiceWorker registered with scope: ', registration.scope);
      }, function(err) {
        console.log('PWA ServiceWorker registration failed: ', err);
      });
    });
  }
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — dark glassy premium theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0d1520 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,20,35,0.95);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0; }

/* ── Main text ── */
h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
p, li, span, label        { color: #cbd5e1; }

/* ── Cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.6);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-safe     { background: rgba(34,197,94,0.18);  color: #4ade80; border: 1px solid #4ade8055; }
.badge-moderate { background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid #fbbf2455; }
.badge-severe   { background: rgba(249,115,22,0.18); color: #fb923c; border: 1px solid #fb923c55; }
.badge-failure  { background: rgba(239,68,68,0.18);  color: #f87171; border: 1px solid #f8717155; }

/* ── Step header ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 18px;
    background: rgba(99,102,241,0.1);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    margin-bottom: 16px;
}
.step-num {
    background: #6366f1;
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    flex-shrink: 0;
}

/* ── RUL progress bars ── */
.rul-bar-wrap { margin-bottom: 10px; }
.rul-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;
}
.rul-bar-bg {
    height: 8px; background: rgba(255,255,255,0.07);
    border-radius: 4px; overflow: hidden;
}
.rul-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #38bdf8);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 12px !important;
    background: rgba(99,102,241,0.04) !important;
    padding: 10px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
}

/* ── Info / warning boxes from streamlit ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LAZY MODEL IMPORTS  (cached so they load only once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resnet(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner=False)
def load_dlm(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner=False)
def load_gru(path):
    from tensorflow.keras.models import load_model as lm
    return lm(path, compile=False)

@st.cache_resource(show_spinner=False)
def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)

@st.cache_resource(show_spinner=False)
def load_scalers(sx_path, sy_path):
    import joblib
    if os.path.exists(sx_path) and os.path.exists(sy_path):
        sx = joblib.load(sx_path)
        sy = joblib.load(sy_path)
        return sx, sy
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(image_path):
    """ResNet50 binary classifier — crack / no-crack."""
    from tensorflow.keras.applications.resnet50 import preprocess_input

    model  = load_resnet(CONFIG["resnet_model"])
    img    = cv2_load(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x      = preprocess_input(
                 np.expand_dims(cv2.resize(img_rgb, (224, 224)), axis=0))
    pred   = float(model.predict(x, verbose=0)[0][0])
    is_crack  = pred >= 0.5
    confidence = pred * 100 if is_crack else (1.0 - pred) * 100

    # figure
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1a1a2e")
    ax.imshow(img_rgb)
    ax.set_title(
        f"{'CRACK DETECTED' if is_crack else 'INTACT — NO CRACK'}  ({confidence:.1f}%)",
        color='#f87171' if is_crack else '#4ade80',
        fontweight='bold', fontsize=11
    )
    ax.axis('off')
    fig.tight_layout()
    return is_crack, confidence, fig


def run_step2(image_path, spacing, roi_length, min_dist):
    """YOLOv8 segmentation + skeleton + perpendicular line grid."""
    from skimage.morphology import skeletonize, binary_closing

    img_gray  = cv2_load(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2_load(image_path, cv2.IMREAD_COLOR)
    if img_gray is None or img_color is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    img_rgb   = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_gray.shape[:2]

    yolo_model   = load_yolo(CONFIG["yolo_model"])
    results      = yolo_model(img_color, verbose=False)

    if not results or results[0].masks is None:
        raise RuntimeError("YOLO found no crack masks in this image.")

    masks_data    = results[0].masks.data.cpu().numpy()
    combined_mask = np.any(masks_data, axis=0).astype(np.uint8) * 255
    binary_mask   = cv2.resize(combined_mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    binary        = binary_mask > 0

    overlay         = img_rgb.copy()
    overlay[binary] = [255, 0, 0]
    img_highlighted = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    binary_closed = binary_closing(binary, np.ones((3, 3)))
    skeleton      = skeletonize(binary_closed)
    yx            = np.column_stack(np.where(skeleton))

    if len(yx) < 10:
        raise RuntimeError("Skeleton too small — crack may be too thin.")

    step    = max(1, len(yx) // max(1, len(yx) // spacing))
    centers = yx[::step]

    filtered = [centers[0]]
    for pt in centers[1:]:
        last = filtered[-1]
        if np.hypot(pt[0] - last[0], pt[1] - last[1]) >= min_dist:
            filtered.append(pt)
    centers = np.array(filtered)

    lines_coords = []
    for i in range(len(centers) - 1):
        p1, p2 = centers[i], centers[i + 1]
        dx, dy = p2[1] - p1[1], p2[0] - p1[0]
        px, py = -dy, dx
        norm   = np.hypot(px, py)
        if norm < 1e-6:
            continue
        px = int(px / norm * roi_length / 2)
        py = int(py / norm * roi_length / 2)
        start = (p1[0] - py, p1[1] - px)
        end   = (p1[0] + py, p1[1] + px)
        lines_coords.append(((start[1], start[0]),
                              (end[1],   end[0]),
                              (p1[1],    p1[0])))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    axes[0].imshow(img_highlighted)
    axes[0].set_title("YOLO Segmentation Mask", color="white", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    for s, e, _ in lines_coords:
        axes[1].plot([s[0], e[0]], [s[1], e[1]],
                     color="cyan", linewidth=0.8, alpha=0.85)
    axes[1].set_title(f"Measurement Grid ({len(lines_coords)} zones)",
                      color="white", fontsize=11)
    axes[1].axis("off")
    fig.tight_layout()

    return img_gray, binary, lines_coords, skeleton, len(centers), fig


def run_step3(img_gray, lines_coords, n_points=501):
    """Extract 501-point perpendicular intensity profiles."""
    from skimage.measure import profile_line
    from scipy.interpolate import interp1d

    profiles, valid_lines = [], []
    for start_pt, end_pt, center_pt in lines_coords:
        src = (start_pt[1], start_pt[0])
        dst = (end_pt[1],   end_pt[0])
        raw = profile_line(img_gray, src, dst, linewidth=1,
                           mode='constant', cval=0)
        if len(raw) < 2:
            continue
        x_old = np.linspace(0, 1, len(raw))
        x_new = np.linspace(0, 1, n_points)
        prof  = interp1d(x_old, raw, kind='linear')(x_new).astype('float32')
        profiles.append(prof)
        valid_lines.append((start_pt, end_pt, center_pt))

    profiles = np.array(profiles)
    return profiles, valid_lines


def run_step4(profiles, pi2mi):
    """DLM Conv1D → crack widths in pixels, µm, mm."""
    dlm   = load_dlm(CONFIG["dlm_model"])
    X     = (profiles / 255.0).reshape(-1, 501, 1)
    w_px  = dlm.predict(X, verbose=0).flatten()
    w_um  = w_px * pi2mi
    w_mm  = w_um / 1000.0
    return w_px, w_um, w_mm, float(np.mean(w_um)), float(np.max(w_um)), \
           float(np.mean(w_mm)), float(np.max(w_mm))


def run_step5(max_um):
    """GRU growth rate + exponential RUL."""
    width = float(max_um)
    csv_path = CONFIG["csv_database"]

    # ── Historical average from CSV ───────────────────────────────────────────
    estimated_prev_width = width * 0.95
    csv_note = "Using 5% baseline (no CSV match)."
    try:
        df     = pd.read_csv(csv_path)
        margin = width * 0.05
        sim    = df[(df["MANwidth_um"] >= width - margin) &
                    (df["MANwidth_um"] <= width + margin)]
        if not sim.empty:
            estimated_prev_width = float(sim["width_prev_um"].mean())
            csv_note = f"Found {len(sim)} historical matches → avg prev width: {estimated_prev_width:.2f} µm"
        else:
            csv_note = "No exact CSV match → using 5% baseline."
    except Exception as exc:
        csv_note = f"CSV error: {exc} → using 5% baseline."

    # ── GRU prediction ────────────────────────────────────────────────────────
    gru_model = load_gru(CONFIG["gru_model"])
    scaler_X, scaler_y = load_scalers(CONFIG["scaler_X"], CONFIG["scaler_y"])

    input_raw = np.array([[estimated_prev_width, width]], dtype=np.float32)

    if scaler_X is not None and scaler_y is not None:
        input_scaled    = scaler_X.transform(input_raw)
        input_gru       = input_scaled.reshape(1, 2, 1)
        growth_scaled   = gru_model.predict(input_gru, verbose=0)
        growth_stage    = float(scaler_y.inverse_transform(growth_scaled)[0][0])
        scaler_note     = "Pre-trained scalers loaded ✓"
    else:
        # Fallback: rough normalise over plausible 0–500 µm range
        x_norm = input_raw / 500.0
        input_gru       = x_norm.reshape(1, 2, 1)
        growth_raw      = float(gru_model.predict(input_gru, verbose=0)[0][0])
        growth_stage    = growth_raw * 50.0         # de-normalise rough estimate
        scaler_note     = "⚠️ scaler_X.pkl / scaler_y.pkl not found — using fallback normalisation"

    days_per_stage   = 28 / 5
    growth_day       = max(growth_stage / days_per_stage, 0.0001)
    k_constant       = growth_day / width

    limits = {100: None, 150: None, 200: None, 300: None}
    for lim in limits:
        limits[lim] = 0.0 if width >= lim else float(np.log(lim / width) / k_constant)

    def classify(w):
        if w < 100:  return "Safe"
        if w < 150:  return "Moderate"
        if w < 200:  return "Severe"
        return "Failure"

    return {
        "width_um"      : width,
        "prev_width_um" : estimated_prev_width,
        "growth_day"    : growth_day,
        "k_constant"    : k_constant,
        "rul_by_limit"  : limits,
        "condition"     : classify(width),
        "csv_note"      : csv_note,
        "scaler_note"   : scaler_note,
    }


def build_report_figure(img_path, img_gray, binary, valid_lines,
                         widths_um, widths_mm, avg_um, max_um,
                         avg_mm, max_mm, rul_data, profiles, skeleton):
    """Return a matplotlib Figure for the final report."""
    img_rgb = cv2.cvtColor(cv2_load(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    condition  = rul_data["condition"]
    k          = rul_data["k_constant"]
    g_day      = rul_data["growth_day"]
    rul_limits = rul_data["rul_by_limit"]
    primary    = rul_limits.get(200, 0)

    prognosis = (
        f"Prognosis: DEGRADING — Failure in ~{primary:.0f} d (200 µm limit)\n"
        f"Max: {max_mm:.4f} mm  |  {max_um:.2f} µm"
    ) if primary > 0 else (
        f"Prognosis: CRITICAL — 200 µm limit exceeded\n"
        f"Max: {max_mm:.4f} mm  |  {max_um:.2f} µm"
    )

    insight = (
        f"💡  Condition = {condition}  │  "
        f"Exp growth k = {k:.5f}/day  │  "
        f"Linear rate = {g_day:.4f} µm/day  │  "
        f"Avg = {avg_mm:.4f} mm  │  Max = {max_mm:.4f} mm"
    )

    fig = plt.figure(figsize=(20, 7), facecolor='#1a1a1a')
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.02, right=0.98, top=0.82, bottom=0.08,
                            wspace=0.04)

    # Panel 1 — YOLO overlay
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor('#1a1a1a')
    ov = img_rgb.copy(); ov[binary] = [220, 30, 30]
    ax1.imshow(cv2.addWeighted(img_rgb, 0.55, ov, 0.45, 0))
    ax1.set_title("YOLO Segmentation", color='white', fontsize=11, pad=6)
    ax1.axis('off')

    # Panel 2 — Grid
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor('#1a1a1a')
    ax2.imshow(img_gray, cmap='gray')
    if skeleton is not None:
        yx_sk = np.column_stack(np.where(skeleton))
        if len(yx_sk):
            ax2.scatter(yx_sk[:, 1], yx_sk[:, 0], c='black', s=0.3, linewidths=0)
    for s, e, _ in valid_lines:
        ax2.plot([s[0], e[0]], [s[1], e[1]], color='cyan', linewidth=0.8, alpha=0.85)
    ax2.set_title("DLM Target Grid", color='white', fontsize=11, pad=6)
    ax2.axis('off')

    # Panel 3 — Profiles
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor('#1a1a1a')
    if profiles is not None and len(profiles):
        for prof in profiles:
            ax3.plot(np.arange(501), prof, color='#aaaaaa', linewidth=0.5, alpha=0.6)
    ax3.set_xlabel("Position (0–500)", color='white', fontsize=9)
    ax3.set_ylabel("Grayscale Intensity", color='white', fontsize=9)
    ax3.tick_params(colors='white', labelsize=8)
    for sp in ax3.spines.values(): sp.set_edgecolor('#444444')
    ax3.set_title("501-pt Intensity Profiles", color='white', fontsize=11, pad=6)

    # RUL table
    rul_lines = ["RUL by limit:"] + [
        f"  {lim} µm → {days:.0f} d" if days > 0 else f"  {lim} µm → EXCEEDED"
        for lim, days in rul_limits.items()
    ]
    fig.text(0.98, 0.97, "\n".join(rul_lines),
             color='red', fontsize=9, fontweight='bold',
             ha='right', va='top', transform=fig.transFigure, linespacing=1.5)

    fig.text(0.98, 0.88, prognosis,
             color='orange', fontsize=9, ha='right', va='top',
             transform=fig.transFigure)

    fig.text(0.01, 0.02, insight,
             color='yellow', fontsize=9, ha='left', va='bottom',
             transform=fig.transFigure)

    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-TUNE YOLO PARAMS
# ══════════════════════════════════════════════════════════════════════════════
def auto_tune(image_path):
    tmp = cv2_load(image_path, cv2.IMREAD_GRAYSCALE)
    if tmp is None:
        return 50, 200, 50
    md = max(tmp.shape)
    if md >= 3000: return 200, 500, 180
    if md >= 800:  return 50,  150, 40
    return 15, 60, 15


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def badge(condition):
    cls = {
        "Safe": "badge-safe", "Moderate": "badge-moderate",
        "Severe": "badge-severe", "Failure": "badge-failure"
    }.get(condition, "badge-moderate")
    return f'<span class="badge {cls}">{condition}</span>'


def rul_bar(limit, days, max_days=730):
    pct   = min(100, 100 * days / max_days) if days > 0 else 0
    label = f"{days:.0f} days" if days > 0 else "EXCEEDED"
    color = "#f87171" if days == 0 else "#fbbf24" if days < 90 else "#4ade80"
    return f"""
<div class="rul-bar-wrap">
  <div class="rul-bar-label">
    <span>{limit} µm limit</span><span style="color:{color}">{label}</span>
  </div>
  <div class="rul-bar-bg">
    <div class="rul-bar-fill" style="width:{pct}%;background:{'linear-gradient(90deg,#f87171,#fb923c)' if days<90 else 'linear-gradient(90deg,#6366f1,#38bdf8)'};"></div>
  </div>
</div>"""


def metric_card(value, label, unit=""):
    return f"""
<div class="metric-card">
  <div class="metric-value">{value}<span style="font-size:1rem;color:#94a3b8"> {unit}</span></div>
  <div class="metric-label">{label}</div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 CrackSense")
    st.markdown("**Structural Health Monitoring**")
    st.markdown("---")

    st.markdown("### ⚙️ Model Paths")
    for name, path in [
        ("ResNet50",  CONFIG["resnet_model"]),
        ("YOLOv8",    CONFIG["yolo_model"]),
        ("DLM",       CONFIG["dlm_model"]),
        ("GRU",       CONFIG["gru_model"]),
    ]:
        exists = os.path.exists(path)
        icon   = "✅" if exists else "❌"
        st.markdown(f"{icon} `{name}` — {'found' if exists else 'NOT FOUND'}")

    scalers_ok = os.path.exists(CONFIG["scaler_X"]) and os.path.exists(CONFIG["scaler_y"])
    st.markdown(f"{'✅' if scalers_ok else '⚠️'} Scalers (pkl) — {'found' if scalers_ok else 'fallback mode'}")

    st.markdown("---")
    st.markdown("### 📐 Pipeline Info")
    st.markdown("""
- **Step 1** ResNet50 — classify  
- **Step 2** YOLOv8 — segment  
- **Step 3** 501-pt profiles  
- **Step 4** DLM — width predict  
- **Step 5** GRU — RUL engine  
""")
    st.markdown("---")
    st.caption("© 2026 CrackSense SHM Dashboard")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 11])
with col_title:
    st.markdown("# 🔬 CrackSense — Structural Health Monitor")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-10px;'>Full CNN pipeline: ResNet50 → "
        "YOLOv8 → 501-pt Profiles → DLM → GRU RUL Engine</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📤 Upload Inspection Image")
uploaded = st.file_uploader(
    "Drag & drop a crack image (JPG / PNG / TIF / BMP)",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
<div style="text-align:center;padding:60px 0;color:#475569;">
  <div style="font-size:3rem;margin-bottom:12px;">📁</div>
  <p style="font-size:1.1rem;">Upload an image above to start the pipeline</p>
  <p style="font-size:0.85rem;color:#334155;">Supported: JPG, PNG, TIF, BMP</p>
</div>""", unsafe_allow_html=True)
    st.stop()

# Save uploaded bytes and keep a reference for reliable OpenCV reading
IMG_BYTES = uploaded.getvalue()
suffix = os.path.splitext(uploaded.name)[-1] or ".tif"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp_f:
    tmp_f.write(IMG_BYTES)
    TMP_PATH = tmp_f.name

# Preview
col_prev, col_info = st.columns([3, 2])
with col_prev:
    st.image(IMG_BYTES, caption=f"Uploaded: {uploaded.name}", width='stretch')
with col_info:
    raw_img = cv2_load(TMP_PATH, cv2.IMREAD_GRAYSCALE)
    if raw_img is not None:
        h, w = raw_img.shape[:2]
        st.markdown(f"""
<div style="padding:16px;background:rgba(255,255,255,0.03);border-radius:12px;border:1px solid rgba(99,102,241,0.2);">
  <p style="color:#94a3b8;margin:0;font-size:0.8rem;">IMAGE INFO</p>
  <p style="margin:6px 0;"><b>Filename:</b> {uploaded.name}</p>
  <p style="margin:6px 0;"><b>Dimensions:</b> {w} × {h} px</p>
  <p style="margin:6px 0;"><b>Size:</b> {uploaded.size/1024:.1f} KB</p>
  <p style="margin:6px 0;"><b>Max dim:</b> {max(h,w)} px</p>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  RUN BUTTON
# ══════════════════════════════════════════════════════════════════════════════
col_btn, col_hint = st.columns([2, 8])
with col_btn:
    run_btn = st.button("🚀  Run Full Pipeline", use_container_width=True)
with col_hint:
    st.markdown(
        "<p style='color:#475569;padding-top:10px;font-size:0.85rem;'>"
        "Models will be cached after first load for faster subsequent runs.</p>",
        unsafe_allow_html=True
    )

if not run_btn:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 Pipeline Results")

# ── Auto-tune params ──────────────────────────────────────────────────────────
spacing, roi_length, min_dist = auto_tune(TMP_PATH)

# ── STEP 1 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">1</div>
  <b>ResNet50 — Crack Classifier</b>
</div>""", unsafe_allow_html=True)

with st.spinner("Running ResNet50 classification…"):
    try:
        is_crack, confidence, fig1 = run_step1(TMP_PATH)
    except Exception as e:
        st.error(f"Step 1 failed: {e}")
        st.stop()

col_s1a, col_s1b = st.columns([3, 2])
with col_s1a:
    st.pyplot(fig1, width='stretch')
    plt.close(fig1)
with col_s1b:
    status_color = "#f87171" if is_crack else "#4ade80"
    status_text  = "CRACK DETECTED" if is_crack else "INTACT — NO CRACK"
    st.markdown(f"""
<div style="padding:24px;background:rgba(255,255,255,0.03);border-radius:14px;
     border:1px solid {status_color}44;text-align:center;margin-top:20px;">
  <div style="font-size:2rem;margin-bottom:8px;">{'🚨' if is_crack else '✅'}</div>
  <div style="color:{status_color};font-size:1.1rem;font-weight:700;">{status_text}</div>
  <div style="color:#94a3b8;font-size:0.85rem;margin-top:8px;">Confidence: {confidence:.2f}%</div>
</div>""", unsafe_allow_html=True)

if not is_crack:
    st.success("✅ No crack detected. Structure appears safe. Pipeline complete.")
    st.stop()

st.markdown("---")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">2</div>
  <b>YOLOv8 — Crack Segmentation & Measurement Grid</b>
</div>""", unsafe_allow_html=True)

with st.spinner("Running YOLOv8 segmentation…"):
    try:
        img_gray, binary, lines_coords, skeleton, n_centers, fig2 = \
            run_step2(TMP_PATH, spacing, roi_length, min_dist)
    except Exception as e:
        st.error(f"Step 2 failed: {e}")
        st.stop()

st.pyplot(fig2, width='stretch')
plt.close(fig2)

cols_s2 = st.columns(3)
cols_s2[0].metric("Sampling spacing",  f"{spacing} px")
cols_s2[1].metric("ROI length",        f"{roi_length} px")
cols_s2[2].metric("Measurement zones", len(lines_coords))

st.markdown("---")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">3</div>
  <b>501-point Intensity Profile Extraction</b>
</div>""", unsafe_allow_html=True)

with st.spinner("Extracting intensity profiles…"):
    profiles, valid_lines = run_step3(img_gray, lines_coords)

if len(profiles) == 0:
    st.warning("⚠️ No valid profiles extracted. Cannot continue.")
    st.stop()

# Plot sample profiles
with st.expander(f"📈 View all {len(profiles)} intensity profiles", expanded=True):
    fig3, ax3 = plt.subplots(figsize=(12, 4), facecolor="#1a1a2e")
    ax3.set_facecolor("#1a1a2e")
    x_axis = np.arange(501)
    for prof in profiles:
        ax3.plot(x_axis, prof, color='#818cf8', linewidth=0.6, alpha=0.5)
    # mean profile highlight
    ax3.plot(x_axis, profiles.mean(axis=0), color='#38bdf8', linewidth=2, label='Mean')
    ax3.set_xlabel("Position (0–500)", color='white', fontsize=9)
    ax3.set_ylabel("Grayscale Intensity", color='white', fontsize=9)
    ax3.tick_params(colors='white', labelsize=8)
    for sp in ax3.spines.values(): sp.set_edgecolor('#333')
    ax3.legend(loc='upper right', labelcolor='white', facecolor='#1a1a2e',
               edgecolor='#444')
    ax3.set_title(f"All {len(profiles)} perpendicular profiles (mean in blue)",
                  color='white', fontsize=10)
    st.pyplot(fig3, width='stretch')
    plt.close(fig3)

st.markdown("---")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">4</div>
  <b>DLM Model — Crack Width Prediction</b>
</div>""", unsafe_allow_html=True)

with st.spinner("DLM predicting crack widths…"):
    try:
        w_px, w_um, w_mm, avg_um, max_um, avg_mm, max_mm = \
            run_step4(profiles, CONFIG["pi2mi"])
    except Exception as e:
        st.error(f"Step 4 failed: {e}")
        st.stop()

# Metric cards
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card(f"{avg_um:.1f}", "Average Width", "µm"), unsafe_allow_html=True)
c2.markdown(metric_card(f"{max_um:.1f}", "Maximum Width", "µm"), unsafe_allow_html=True)
c3.markdown(metric_card(f"{avg_mm:.4f}", "Average Width", "mm"), unsafe_allow_html=True)
c4.markdown(metric_card(f"{max_mm:.4f}", "Maximum Width", "mm"), unsafe_allow_html=True)

# Width distribution bar chart
with st.expander("📊 Width distribution across measurement zones"):
    fig4, ax4 = plt.subplots(figsize=(12, 3.5), facecolor="#1a1a2e")
    ax4.set_facecolor("#1a1a2e")
    zone_ids = np.arange(len(w_um))
    colors = ['#f87171' if v >= 200 else '#fbbf24' if v >= 100 else '#4ade80'
              for v in w_um]
    ax4.bar(zone_ids, w_um, color=colors, width=0.8)
    ax4.axhline(avg_um, color='#818cf8', linewidth=1.5, linestyle='--', label='Mean')
    ax4.set_xlabel("Zone index", color='white', fontsize=9)
    ax4.set_ylabel("Width (µm)", color='white', fontsize=9)
    ax4.tick_params(colors='white', labelsize=8)
    for sp in ax4.spines.values(): sp.set_edgecolor('#333')
    ax4.legend(labelcolor='white', facecolor='#1a1a2e', edgecolor='#444')
    ax4.set_title("Per-zone crack widths (green < 100 µm, yellow < 200, red ≥ 200)",
                  color='white', fontsize=9)
    st.pyplot(fig4, width='stretch')
    plt.close(fig4)

st.markdown("---")

# ── STEP 5 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">5</div>
  <b>GRU Engine — Growth Rate & Remaining Useful Life</b>
</div>""", unsafe_allow_html=True)

with st.spinner("GRU RUL engine running…"):
    try:
        rul_data = run_step5(max_um)
    except Exception as e:
        st.error(f"Step 5 failed: {e}")
        st.stop()

st.info(f"**CSV lookup:** {rul_data['csv_note']}")
if "fallback" in rul_data['scaler_note'].lower() or "⚠️" in rul_data['scaler_note']:
    st.warning(rul_data['scaler_note'])
else:
    st.success(rul_data['scaler_note'])

# Main RUL display
col_rul_l, col_rul_r = st.columns([3, 2])

with col_rul_l:
    st.markdown("#### ⏳ Remaining Useful Life by Limit")
    for lim, days in rul_data["rul_by_limit"].items():
        st.markdown(rul_bar(lim, days), unsafe_allow_html=True)

with col_rul_r:
    cond = rul_data["condition"]
    st.markdown("#### 🩺 Condition Assessment")
    st.markdown(f"""
<div style="padding:20px;background:rgba(255,255,255,0.04);border-radius:14px;
     border:1px solid rgba(99,102,241,0.25);text-align:center;margin-bottom:12px;">
  <div style="font-size:1.8rem;margin-bottom:8px;">
    {'✅' if cond=='Safe' else '⚠️' if cond=='Moderate' else '🔶' if cond=='Severe' else '🚨'}
  </div>
  {badge(cond)}
  <div style="margin-top:14px;color:#94a3b8;font-size:0.82rem;">
    Current width: <b style="color:#e2e8f0">{rul_data['width_um']:.2f} µm</b>
  </div>
</div>
<div style="padding:14px;background:rgba(255,255,255,0.03);border-radius:10px;
     border:1px solid rgba(99,102,241,0.15);font-size:0.82rem;color:#94a3b8;">
  <div>Growth rate: <b style="color:#e2e8f0">{rul_data['growth_day']:.4f} µm/day</b></div>
  <div>Exp. k: <b style="color:#e2e8f0">{rul_data['k_constant']:.6f} /day</b></div>
  <div>Daily increase: <b style="color:#e2e8f0">{rul_data['k_constant']*100:.3f}%/day</b></div>
  <div>Prev. est. width: <b style="color:#e2e8f0">{rul_data['prev_width_um']:.2f} µm</b></div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📄 Final Analysis Report")

with st.spinner("Generating final report figure…"):
    report_fig = build_report_figure(
        TMP_PATH, img_gray, binary, valid_lines,
        w_um, w_mm, avg_um, max_um, avg_mm, max_mm,
        rul_data, profiles, skeleton
    )

st.pyplot(report_fig, width='stretch')

# Download button
report_bytes = fig_to_bytes(report_fig)
plt.close(report_fig)

st.download_button(
    label="⬇️  Download Report (PNG)",
    data=report_bytes,
    file_name="crack_analysis_report.png",
    mime="image/png",
)

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Run Summary")

summary_data = {
    "Metric": [
        "Image", "Crack Detected", "ResNet Confidence",
        "Measurement zones", "Avg Width", "Max Width",
        "Condition", "Growth Rate", "RUL (100 µm)", "RUL (150 µm)",
        "RUL (200 µm)", "RUL (300 µm)",
    ],
    # All values cast to str to avoid Arrow/PyArrow mixed-type serialisation error
    "Value": [
        str(uploaded.name),
        "YES ⚠️" if is_crack else "NO ✅",
        f"{confidence:.2f}%",
        str(len(valid_lines)),
        f"{avg_um:.2f} µm  ({avg_mm:.4f} mm)",
        f"{max_um:.2f} µm  ({max_mm:.4f} mm)",
        str(rul_data["condition"]),
        f"{rul_data['growth_day']:.4f} µm/day",
        f"{rul_data['rul_by_limit'][100]:.0f} days" if rul_data['rul_by_limit'][100] > 0 else "EXCEEDED",
        f"{rul_data['rul_by_limit'][150]:.0f} days" if rul_data['rul_by_limit'][150] > 0 else "EXCEEDED",
        f"{rul_data['rul_by_limit'][200]:.0f} days" if rul_data['rul_by_limit'][200] > 0 else "EXCEEDED",
        f"{rul_data['rul_by_limit'][300]:.0f} days" if rul_data['rul_by_limit'][300] > 0 else "EXCEEDED",
    ]
}
st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8rem;'>"
    "CrackSense SHM Dashboard — Powered by ResNet50, YOLOv8, DLM & GRU</p>",
    unsafe_allow_html=True
)

# Clean up temp file
try:
    os.unlink(TMP_PATH)
except Exception:
    pass
