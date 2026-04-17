"""
==============================================================================
  FULL CNN CRACK DETECTION & STRUCTURAL HEALTH MONITORING PIPELINE
  Integrates: ResNet50 → YOLO → 501-pt Profile → DLM → LSTM RUL Engine
==============================================================================
"""

# ── Standard Libraries ────────────────────────────────────────────────────────
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Deep Learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# ── Image Processing ──────────────────────────────────────────────────────────
from skimage.measure import profile_line
from skimage.morphology import skeletonize, binary_closing
from scipy.interpolate import interp1d

# ── YOLO ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


# ==============================================================================
# CONFIGURATION  ←  Edit these paths before running
# ==============================================================================
# ==============================================================================
# CONFIGURATION  ←  Edit these paths before running
# ==============================================================================
CONFIG = {
    "image_path"      : "/content/drive/MyDrive/crackdetection/CMd_0.23_20mths_Image4.tif",
    "resnet_model"    : "/content/drive/MyDrive/crackdetection/resnet_crack_classifier.keras",
    "yolo_model"      : "/content/drive/MyDrive/crackdetection/best.pt",
    "dlm_model"       : "/content/drive/MyDrive/crackdetection/krknet_replica.keras",

    # NEW: Paths for the GRU Engine and the CSV Database for historical averaging
    "gru_model"       : "gru_growth_model.keras", 
    "csv_database"    : "/content/drive/MyDrive/crackdetection/krkCMd_table_with_growth.csv",

    "pi2mi"           : 1000 * 25.4 / 6400,
    "spacing"         : 100,
    "roi_length"      : 500,
    "min_dist"        : 180,
}


# ==============================================================================
# STEP 1 ─ ResNet50: Does a crack exist?
# ==============================================================================
def step1_classify(image_path, model_path):
    print("\n" + "="*60)
    print("  STEP 1 │ ResNet50 Crack Classifier")
    print("="*60)

    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    x          = preprocess_input(np.expand_dims(img_resized, axis=0))

    prediction = model.predict(x, verbose=0)[0][0]
    is_crack   = prediction >= 0.5
    confidence = prediction * 100 if is_crack else (1.0 - prediction) * 100
    status     = "CRACK DETECTED" if is_crack else "INTACT (NO CRACK)"

    print(f"  Result     : {status}")
    print(f"  Confidence : {confidence:.2f}%")

    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"{status}  ({confidence:.1f}%)",
              color='red' if is_crack else 'green', fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return is_crack


# ==============================================================================
# STEP 2 ─ YOLO: Segment crack + build perpendicular measurement grid
# ==============================================================================
def step2_segment(image_path, yolo_path, spacing, roi_length, min_dist):
    print("\n" + "="*60)
    print("  STEP 2 │ YOLOv8 Segmentation & Centerline Extraction")
    print("="*60)

    img_gray  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path)
    img_rgb   = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_gray.shape[:2]

    print("  Running YOLOv8 segmentation...")
    yolo_model  = YOLO(yolo_path)
    results     = yolo_model(img_color, verbose=False)

    if not results or results[0].masks is None:
        raise RuntimeError("YOLO found no crack masks in this image.")

    masks_data    = results[0].masks.data.cpu().numpy()
    combined_mask = np.any(masks_data, axis=0).astype(np.uint8) * 255
    binary_mask   = cv2.resize(combined_mask, (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
    binary        = binary_mask > 0
    print("  YOLO segmentation successful.")

    overlay         = img_rgb.copy()
    overlay[binary] = [255, 0, 0]
    img_highlighted = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    print("  Extracting skeleton and centerline points...")
    binary_closed = binary_closing(binary, np.ones((3, 3)))
    skeleton      = skeletonize(binary_closed)

    yx = np.column_stack(np.where(skeleton))
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
    print(f"  Mapped {len(centers)} centerline sample points.")

    lines_coords = []
    for i in range(len(centers) - 1):
        p1   = centers[i]
        p2   = centers[i + 1]
        dx   = p2[1] - p1[1]
        dy   = p2[0] - p1[0]
        px, py = -dy, dx
        norm = np.hypot(px, py)
        if norm < 1e-6:
            continue
        px = int(px / norm * roi_length / 2)
        py = int(py / norm * roi_length / 2)
        start = (p1[0] - py, p1[1] - px)
        end   = (p1[0] + py, p1[1] + px)
        lines_coords.append(((start[1], start[0]),
                              (end[1],   end[0]),
                              (p1[1],    p1[0])))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("YOLO Segmentation Mask")
    plt.imshow(img_highlighted); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Measurement Grid ({len(lines_coords)} zones)")
    plt.imshow(img_gray, cmap='gray')
    for s, e, _ in lines_coords:
        plt.plot([s[0], e[0]], [s[1], e[1]],
                 color='cyan', linewidth=1, alpha=0.8)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return img_gray, binary, lines_coords, skeleton


# ==============================================================================
# STEP 3 ─ Extract 501-point intensity profiles along perpendicular lines
# ==============================================================================
def step3_extract_profiles(img_gray, lines_coords, n_points=501):
    print("\n" + "="*60)
    print("  STEP 3 │ Extracting 501-point Intensity Profiles")
    print("="*60)

    profiles    = []
    valid_lines = []

    for start_pt, end_pt, center_pt in lines_coords:
        src = (start_pt[1], start_pt[0])
        dst = (end_pt[1],   end_pt[0])

        raw_profile = profile_line(img_gray, src, dst, linewidth=1,
                                   mode='constant', cval=0)

        if len(raw_profile) < 2:
            continue

        x_old  = np.linspace(0, 1, len(raw_profile))
        x_new  = np.linspace(0, 1, n_points)
        interp = interp1d(x_old, raw_profile, kind='linear')
        profile_501 = interp(x_new).astype('float32')

        profiles.append(profile_501)
        valid_lines.append((start_pt, end_pt, center_pt))

    profiles = np.array(profiles)
    print(f"  Extracted {len(profiles)} valid 501-point profiles.")
    return profiles, valid_lines


# ==============================================================================
# STEP 4 ─ DLM Model: Predict crack widths from profiles
# ==============================================================================
def step4_predict_widths(profiles, dlm_model_path, pi2mi):
    print("\n" + "="*60)
    print("  STEP 4 │ DLM Model → Predicting Crack Widths")
    print("="*60)

    dlm_model = tf.keras.models.load_model(dlm_model_path)

    # Normalise (0-255 → 0-1) and reshape for Conv1D: (N, 501, 1)
    X = (profiles / 255.0).reshape(-1, 501, 1)

    widths_pixels = dlm_model.predict(X, verbose=0).flatten()

    # Convert pixels → micrometers → millimeters
    widths_um = widths_pixels * pi2mi
    widths_mm = widths_um / 1000.0

    avg_um = float(np.mean(widths_um))
    max_um = float(np.max(widths_um))
    avg_mm = avg_um / 1000.0
    max_mm = max_um / 1000.0

    print(f"  Profiles predicted : {len(widths_mm)}")
    print(f"  Average width      : {avg_um:.2f} µm  ({avg_mm:.4f} mm)")
    print(f"  Maximum width      : {max_um:.2f} µm  ({max_mm:.4f} mm)")

    return widths_pixels, widths_um, widths_mm, avg_um, max_um, avg_mm, max_mm


# ==============================================================================
# STEP 5 ─ LSTM RUL Engine (replaces old severity + linear/exponential RUL)
#   Input  : max_um  — maximum crack width in micrometers from Step 4
#   Output : predicted growth rate, exponential RUL vs each limit
# ==============================================================================
# Change the function name
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==============================================================================
# STEP 5 ─ GRU RUL Engine (With Database Historical Averaging)
# ==============================================================================
def step5_gru_rul(max_um, gru_model_path, csv_path):
    print("\n" + "="*60)
    print("  STEP 5 │ GRU Exponential RUL Engine (Sequence Averaging)")
    print("="*60)

    width = float(max_um)
    print(f"  Current measured width (max) : {width:.2f} µm")

    # ── 1. Look into the CSV and take the average of previous widths ───────────
    print(f"  Querying database for historical average...")
    try:
        df = pd.read_csv(csv_path)
        margin = width * 0.05 # Look for cracks within +/- 5% of our current width
        similar_cracks = df[(df["MANwidth_um"] >= width - margin) & (df["MANwidth_um"] <= width + margin)]
        
        if not similar_cracks.empty:
            estimated_prev_width = similar_cracks["width_prev_um"].mean()
            print(f"  -> Found {len(similar_cracks)} similar cracks. Averaged past width: {estimated_prev_width:.2f} µm")
        else:
            estimated_prev_width = width * 0.95
            print(f"  -> No exact matches. Using 5% baseline: {estimated_prev_width:.2f} µm")
    except Exception as e:
        print(f"  -> Warning: Could not read CSV ({e}). Using 5% baseline.")
        estimated_prev_width = width * 0.95

    # ── 2. Load the pre-trained GRU model and Scalers ──────────────────────────
    if not os.path.exists(gru_model_path):
        raise FileNotFoundError(f"GRU model not found at '{gru_model_path}'.")
    
    model_growth = load_model(gru_model_path)
    
    # Load the scalers you saved during training so the GRU understands the input
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # ── 3. Prepare the Sequence Input [Past, Present] ──────────────────────────
    input_raw = np.array([[estimated_prev_width, width]])
    input_scaled = scaler_X.transform(input_raw)
    input_gru = input_scaled.reshape(1, 2, 1)

    days_per_stage = 28 / 5   # 5.6 days per stage interval

    # ── 4. GRU Predicts instantaneous growth rate ──────────────────────────────
    growth_pred_scaled = model_growth.predict(input_gru, verbose=0)
    growth_pred_stage = scaler_y.inverse_transform(growth_pred_scaled)[0][0]
    
    growth_pred_day = growth_pred_stage / days_per_stage
    print(f"  Predicted linear velocity    : +{growth_pred_day:.4f} µm/day")

    # ── 5. Exponential physics engine ──────────────────────────────────────────
    safe_growth_rate = max(growth_pred_day, 0.0001)
    k_constant = safe_growth_rate / width   # dW/dt = W*k  →  k = (dW/dt)/W

    print(f"  Exponential growth rate k    : +{k_constant:.6f} per day")
    print(f"  Exponential daily increase   : +{k_constant * 100:.2f}% per day")
    print("-" * 50)

    limits = [100, 150, 200, 300]
    rul_results = {}

    print("  Remaining Useful Life (Exponential):")
    for limit in limits:
        if width >= limit:
            rul_days = 0.0
            print(f"    RUL ({limit:>3} µm limit): {rul_days:.2f} days  ← LIMIT EXCEEDED")
        else:
            rul_days = np.log(limit / width) / k_constant
            print(f"    RUL ({limit:>3} µm limit): {rul_days:.2f} days")
        rul_results[limit] = rul_days

    def classify_condition(w):
        if w < 100:  return "Safe"
        elif w < 150: return "Moderate"
        elif w < 200: return "Severe"
        else:         return "Failure"

    condition = classify_condition(width)
    print(f"\n  Condition Level            : {condition}")

    return {
        "width_um"          : width,
        "growth_rate_day"   : growth_pred_day,
        "k_constant"        : k_constant,
        "rul_by_limit"      : rul_results,
        "condition"         : condition,
    }


# ==============================================================================
# FINAL REPORT ─ Visualization
#   Dark background │ 3-panel top row │ INSIGHT bar bottom │ Prognosis top-right
# ==============================================================================
def generate_final_report(image_path, img_gray, binary, valid_lines,
                           widths_um, widths_mm,
                           avg_um, max_um, avg_mm, max_mm,
                           rul_data, profiles=None, skeleton=None):

    print("\n" + "="*60)
    print("  FINAL REPORT │ Generating Visualization")
    print("="*60)

    img_color = cv2.imread(image_path)
    img_rgb   = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    condition  = rul_data["condition"]
    k_constant = rul_data["k_constant"]
    growth_day = rul_data["growth_rate_day"]

    # ── Build prognosis & insight strings from LSTM RUL data ──────────────────
    rul_by_limit = rul_data["rul_by_limit"]

    # Use the 200 µm limit as the primary RUL reference for display
    primary_limit = 200
    primary_rul   = rul_by_limit.get(primary_limit, 0)

    if primary_rul > 0:
        prognosis_text = (
            f"Prognosis: DEGRADING — Failure in ~{primary_rul:.0f} days "
            f"({primary_limit} µm limit)\n"
            f"(Max Width Detected: {max_mm:.4f} mm  |  {max_um:.2f} µm)"
        )
    else:
        prognosis_text = (
            f"Prognosis: CRITICAL — {primary_limit} µm limit already exceeded\n"
            f"(Max Width Detected: {max_mm:.4f} mm  |  {max_um:.2f} µm)"
        )

    insight_text = (
        f"💡 INSIGHT: Condition = {condition}  │  "
        f"Exp growth k = {k_constant:.5f}/day  │  "
        f"Linear rate = {growth_day:.4f} µm/day  │  "
        f"Avg = {avg_mm:.4f} mm  │  Max = {max_mm:.4f} mm"
    )

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 7), facecolor='#1a1a1a')
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.02, right=0.98, top=0.82, bottom=0.08,
        wspace=0.04
    )

    # ── PANEL 1: YOLO Segmentation (red crack overlay) ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a1a')
    overlay         = img_rgb.copy()
    overlay[binary] = [220, 30, 30]
    img_highlighted = cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0)
    ax1.imshow(img_highlighted)
    ax1.set_title("YOLO Segmentation", color='white', fontsize=11, pad=6)
    ax1.axis('off')

    # ── PANEL 2: Grayscale + centerline + cyan perpendicular lines ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a1a')
    ax2.imshow(img_gray, cmap='gray')
    if skeleton is not None:
        yx_skel = np.column_stack(np.where(skeleton))
        if len(yx_skel) > 0:
            ax2.scatter(yx_skel[:, 1], yx_skel[:, 0],
                        c='black', s=0.3, linewidths=0)
    for s, e, _ in valid_lines:
        ax2.plot([s[0], e[0]], [s[1], e[1]],
                 color='cyan', linewidth=0.8, alpha=0.85)
    ax2.set_title("DLM Target Grid", color='white', fontsize=11, pad=6)
    ax2.axis('off')

    # ── PANEL 3: All 501-pt profiles overlaid ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#1a1a1a')
    if profiles is not None and len(profiles) > 0:
        x_axis = np.arange(501)
        for prof in profiles:
            ax3.plot(x_axis, prof, color='#aaaaaa', linewidth=0.5, alpha=0.6)
    ax3.set_xlabel("Position (0-500)", color='white', fontsize=9)
    ax3.set_ylabel("Grayscale Intensity", color='white', fontsize=9)
    ax3.tick_params(colors='white', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444444')
    ax3.set_title("501-pt Intensity Profiles", color='white', fontsize=11, pad=6)

    # ── RUL table — rendered as a small text block top-right ──────────────────
    rul_lines = ["RUL by limit:"] + [
        f"  {lim} µm → {days:.0f} d" if days > 0
        else f"  {lim} µm → EXCEEDED"
        for lim, days in rul_by_limit.items()
    ]
    fig.text(0.98, 0.97, "\n".join(rul_lines),
             color='red', fontsize=9, fontweight='bold',
             ha='right', va='top',
             transform=fig.transFigure,
             linespacing=1.5)

    # ── Prognosis text — below RUL table ──────────────────────────────────────
    fig.text(0.98, 0.88, prognosis_text,
             color='orange', fontsize=9,
             ha='right', va='top',
             transform=fig.transFigure)

    # ── INSIGHT bar — bottom ──────────────────────────────────────────────────
    fig.text(0.01, 0.02, insight_text,
             color='yellow', fontsize=9,
             ha='left', va='bottom',
             transform=fig.transFigure)

    plt.savefig("crack_report.png", dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.show()
    print("  Report saved → crack_report.png")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_pipeline(cfg):
    print("\n" + "="*60)
    print("  CNN CRACK DETECTION & STRUCTURAL HEALTH PIPELINE")
    print("="*60)

    image_path = cfg["image_path"]

    # ── Auto-tune YOLO params based on image size ─────────────────────────────
    temp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if temp is not None:
        max_dim = max(temp.shape)
        if max_dim >= 3000:
            cfg["spacing"], cfg["roi_length"], cfg["min_dist"] = 200, 500, 180
        elif max_dim >= 800:
            cfg["spacing"], cfg["roi_length"], cfg["min_dist"] = 50, 150, 40
        else:
            cfg["spacing"], cfg["roi_length"], cfg["min_dist"] = 15, 60, 15

    # ── STEP 1: Classify ──────────────────────────────────────────────────────
    is_crack = step1_classify(image_path, cfg["resnet_model"])

    if not is_crack:
        print("\n✅  No crack detected. Pipeline complete — structure is safe.")
        return

    # ── STEP 2: Segment ───────────────────────────────────────────────────────
    img_gray, binary, lines_coords, skeleton = step2_segment(
        image_path, cfg["yolo_model"],
        cfg["spacing"], cfg["roi_length"], cfg["min_dist"]
    )

    # ── STEP 3: Extract profiles ──────────────────────────────────────────────
    profiles, valid_lines = step3_extract_profiles(img_gray, lines_coords)

    if len(profiles) == 0:
        print("\n⚠️  No valid profiles could be extracted. Exiting.")
        return

    # ── STEP 4: Predict widths ────────────────────────────────────────────────
    widths_pixels, widths_um, widths_mm, avg_um, max_um, avg_mm, max_mm = \
        step4_predict_widths(profiles, cfg["dlm_model"], cfg["pi2mi"])

    # ── STEP 5: LSTM RUL — uses MAX width (closest point to failure) ──────────
    #   max_um is passed directly: the DLM predicted pixel width × pi2mi
    # Change this:

    # To this:
    # ── STEP 5: GRU RUL — uses MAX width (closest point to failure) ──────────
    rul_data = step5_gru_rul(max_um, cfg["gru_model"], cfg["csv_database"])

    # ── Final Report ──────────────────────────────────────────────────────────
    generate_final_report(
        image_path, img_gray, binary, valid_lines,
        widths_um, widths_mm,
        avg_um, max_um, avg_mm, max_mm,
        rul_data,
        profiles=profiles,
        skeleton=skeleton
    )

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_pipeline(CONFIG)