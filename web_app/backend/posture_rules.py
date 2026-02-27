"""
posture_rules.py
─────────────────────────────────────────────────────────────────────────────
Fully deterministic postural screening engine.

Input : MediaPipe Pose 33-landmark list (normalised x, y, z, visibility).
Output: Dict with detected conditions, severity levels, raw angles, and
        annotated landmark indices for SVG skeleton overlay.

No model weights — pure trigonometry & configurable clinical thresholds.
"""

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- Calibration Offsets ---
# MediaPipe Pose landmarks for ears/shoulders often result in a non-zero angle 
# even when a person feels they are "straight". This offset aligns the 
# clinical "Normal" with the expected user baseline.
CALIBRATION = {
    "fhp_offset": 10.0, # degrees
}

# ─── MediaPipe landmark indices ──────────────────────────────────────────────
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM = {
    "nose":          0,
    "l_eye":         2,   "r_eye":         5,
    "l_ear":         7,   "r_ear":         8,
    "l_shoulder":    11,  "r_shoulder":    12,
    "l_elbow":       13,  "r_elbow":       14,
    "l_wrist":       15,  "r_wrist":       16,
    "l_hip":         23,  "r_hip":         24,
    "l_knee":        25,  "r_knee":        26,
    "l_ankle":       27,  "r_ankle":       28,
    "l_heel":        29,  "r_heel":        30,
    "l_foot_index":  31,  "r_foot_index":  32,
}

# ─── Configurable clinical thresholds ────────────────────────────────────────
# All angles are in degrees unless noted.
THRESHOLDS = {
    # Forward Head Posture: Ear offset ahead of shoulder (saggital)
    "fhp_mild":      15.0,   # deg
    "fhp_moderate":  22.0,
    "fhp_severe":    30.0,

    # Kyphosis: Neck-Shoulder-Hip angle (thoracic curvature proxy).
    # Ideal ≈ 170°. Lower → more hunched.
    "kyphosis_mild":      162.0,
    "kyphosis_moderate":  150.0,
    "kyphosis_severe":    138.0,

    # Lordosis: Shoulder-Hip-Knee angle (lumbar curve proxy).
    # Ideal ≈ 172°. Lower → more sway.
    "lordosis_mild":      165.0,
    "lordosis_moderate":  155.0,
    "lordosis_severe":    145.0,

    # Pelvic Tilt: Vertical asymmetry of hips (deg).
    "pelvic_mild":    2.0,
    "pelvic_moderate":4.0,
    "pelvic_severe":  7.0,

    # Scoliosis Indicator: Combined shoulder + hip lateral asymmetry (deg).
    "scoliosis_mild":     3.0,
    "scoliosis_moderate": 6.0,
    "scoliosis_severe":   10.0,

    # Genu Valgum / Varum: Knee-gap / ankle-gap ratio.
    # Valgum (knock-knee):  ratio > threshold  → knees closer than ankles
    # Varum  (bow-leg):     ratio < threshold  → knees wider than ankles
    "valgum_mild":      0.80,  # knee_gap / ankle_gap ratio thresholds
    "valgum_moderate":  0.65,
    "valgum_severe":    0.50,
    "varum_mild":       1.25,
    "varum_moderate":   1.45,
    "varum_severe":     1.70,

    # Visibility gate – skip landmark if below this confidence
    "visibility_min":   0.4,
}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _pt(lms: list[dict], key: str) -> dict | None:
    """Return landmark dict for a named key, or None if not visible enough."""
    idx = LM.get(key)
    if idx is None or idx >= len(lms):
        return None
    lm = lms[idx]
    if lm.get("visibility", 0) < THRESHOLDS["visibility_min"]:
        return None
    return lm


def _angle_3pts(a: dict, b: dict, c: dict) -> float:
    """
    Angle at vertex B formed by rays B→A and B→C. Returns degrees [0, 180].
    Uses dot-product formula for numerical stability.
    """
    ax, ay = a["x"] - b["x"], a["y"] - b["y"]
    cx, cy = c["x"] - b["x"], c["y"] - b["y"]
    dot = ax * cx + ay * cy
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a < 1e-9 or mag_c < 1e-9:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_val))


def _vertical_angle(a: dict, b: dict) -> float:
    """Angle (degrees) between line A→B and true vertical (downward)."""
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    if abs(dy) < 1e-9 and abs(dx) < 1e-9:
        return 0.0
    # Vertical reference vector is (0, 1) in image coords (y grows down)
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def _midpoint(a: dict, b: dict) -> dict:
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "visibility": min(a.get("visibility", 1), b.get("visibility", 1))}


def _severity_from_value(value: float, mild: float, moderate: float, severe: float, higher_is_worse: bool = True) -> str:
    """Map a measured value to a severity string. Direction configurable."""
    if higher_is_worse:
        if value >= severe:   return "Severe"
        if value >= moderate: return "Moderate"
        if value >= mild:     return "Mild"
        return "Normal"
    else:  # lower is worse (angles like kyphosis)
        if value <= severe:   return "Severe"
        if value <= moderate: return "Moderate"
        if value <= mild:     return "Mild"
        return "Normal"


# ─── Individual condition analysers ──────────────────────────────────────────

def _check_forward_head(lms: list[dict]) -> dict:
    """Forward Head Posture — ear offset relative to shoulder."""
    result = {"condition": "Forward Head Posture", "code": "FHP"}
    
    ear_l, ear_r   = _pt(lms, "l_ear"), _pt(lms, "r_ear")
    sh_l,  sh_r    = _pt(lms, "l_shoulder"), _pt(lms, "r_shoulder")

    # Prefer same side; both sides if available → average
    pairs = []
    if ear_l and sh_l: pairs.append((ear_l, sh_l))
    if ear_r and sh_r: pairs.append((ear_r, sh_r))

    if not pairs:
        return {**result, "severity": "Unknown", "angle": None, "note": "Ear/shoulder not visible"}

    angles = [_vertical_angle(sh, ear) for sh, ear in pairs]
    raw_angle  = sum(angles) / len(angles)
    
    # Apply calibration offset
    angle = max(0.0, raw_angle - CALIBRATION["fhp_offset"])

    severity = _severity_from_value(angle,
        THRESHOLDS["fhp_mild"], THRESHOLDS["fhp_moderate"], THRESHOLDS["fhp_severe"])

    return {**result, "severity": severity, "angle": round(angle, 1), "unit": "deg",
            "ideal": "< 15°", "measured": f"{angle:.1f}°",
            "landmarks": [LM["l_ear"], LM["r_ear"], LM["l_shoulder"], LM["r_shoulder"]]}


def _check_kyphosis(lms: list[dict]) -> dict:
    """Kyphosis — thoracic curve via neck–shoulder–hip angle."""
    result = {"condition": "Kyphosis (Hunched Back)", "code": "KYPHOSIS"}

    # Approximate neck = midpoint of shoulders elevated by ~head height
    sh_l, sh_r = _pt(lms, "l_shoulder"), _pt(lms, "r_shoulder")
    hip_l, hip_r = _pt(lms, "l_hip"),    _pt(lms, "r_hip")
    nose = _pt(lms, "nose")

    if not (sh_l and sh_r and hip_l and hip_r and nose):
        return {**result, "severity": "Unknown", "angle": None, "note": "Required landmarks not visible"}

    mid_sh  = _midpoint(sh_l, sh_r)
    mid_hip = _midpoint(hip_l, hip_r)

    # Neck proxy = midpoint between shoulders and nose (top of head–shoulder region)
    neck = _midpoint(mid_sh, nose)

    angle = _angle_3pts(neck, mid_sh, mid_hip)

    severity = _severity_from_value(angle,
        THRESHOLDS["kyphosis_mild"], THRESHOLDS["kyphosis_moderate"], THRESHOLDS["kyphosis_severe"],
        higher_is_worse=False)

    return {**result, "severity": severity, "angle": round(angle, 1), "unit": "deg",
            "ideal": "> 162°", "measured": f"{angle:.1f}°",
            "landmarks": [LM["nose"], LM["l_shoulder"], LM["r_shoulder"], LM["l_hip"], LM["r_hip"]]}


def _check_lordosis(lms: list[dict]) -> dict:
    """Lordosis — lumbar swayback via shoulder–hip–knee angle."""
    result = {"condition": "Lordosis (Swayback)", "code": "LORDOSIS"}

    sh_l, sh_r   = _pt(lms, "l_shoulder"), _pt(lms, "r_shoulder")
    hip_l, hip_r = _pt(lms, "l_hip"),      _pt(lms, "r_hip")
    kn_l, kn_r   = _pt(lms, "l_knee"),     _pt(lms, "r_knee")

    if not (sh_l and sh_r and hip_l and hip_r):
        return {**result, "severity": "Unknown", "angle": None, "note": "Required landmarks not visible"}

    mid_sh  = _midpoint(sh_l, sh_r)
    mid_hip = _midpoint(hip_l, hip_r)

    angles = []
    if kn_l:  angles.append(_angle_3pts(mid_sh, mid_hip, kn_l))
    if kn_r:  angles.append(_angle_3pts(mid_sh, mid_hip, kn_r))

    if not angles:
        return {**result, "severity": "Unknown", "angle": None, "note": "Knees not visible"}

    angle = sum(angles) / len(angles)

    severity = _severity_from_value(angle,
        THRESHOLDS["lordosis_mild"], THRESHOLDS["lordosis_moderate"], THRESHOLDS["lordosis_severe"],
        higher_is_worse=False)

    return {**result, "severity": severity, "angle": round(angle, 1), "unit": "deg",
            "ideal": "> 165°", "measured": f"{angle:.1f}°",
            "landmarks": [LM["l_shoulder"], LM["r_shoulder"], LM["l_hip"], LM["r_hip"], LM["l_knee"], LM["r_knee"]]}


def _check_pelvic_tilt(lms: list[dict]) -> dict:
    """Pelvic Tilt — vertical asymmetry between left and right hip."""
    result = {"condition": "Pelvic Tilt", "code": "PELVIC_TILT"}

    hip_l, hip_r = _pt(lms, "l_hip"), _pt(lms, "r_hip")

    if not (hip_l and hip_r):
        return {**result, "severity": "Unknown", "angle": None, "note": "Hips not visible"}

    # Angle from horizontal
    dy = hip_r["y"] - hip_l["y"]
    dx = hip_r["x"] - hip_l["x"]
    angle = abs(math.degrees(math.atan2(abs(dy), max(abs(dx), 1e-9))))

    severity = _severity_from_value(angle,
        THRESHOLDS["pelvic_mild"], THRESHOLDS["pelvic_moderate"], THRESHOLDS["pelvic_severe"])

    tilt_side = "Right hip higher" if hip_r["y"] < hip_l["y"] else "Left hip higher"

    return {**result, "severity": severity, "angle": round(angle, 1), "unit": "deg",
            "ideal": "< 2°", "measured": f"{angle:.1f}°", "note": tilt_side if severity != "Normal" else "",
            "landmarks": [LM["l_hip"], LM["r_hip"]]}


def _check_scoliosis(lms: list[dict]) -> dict:
    """Scoliosis Indicator — combined lateral asymmetry of shoulders and hips."""
    result = {"condition": "Scoliosis Indicator", "code": "SCOLIOSIS"}

    sh_l, sh_r   = _pt(lms, "l_shoulder"), _pt(lms, "r_shoulder")
    hip_l, hip_r = _pt(lms, "l_hip"),      _pt(lms, "r_hip")

    if not (sh_l and sh_r and hip_l and hip_r):
        return {**result, "severity": "Unknown", "angle": None, "note": "Shoulder/hip landmarks not visible"}

    # Shoulder tilt angle
    sh_dy  = abs(sh_r["y"]  - sh_l["y"])
    sh_dx  = abs(sh_r["x"]  - sh_l["x"])
    sh_ang = math.degrees(math.atan2(sh_dy, max(sh_dx, 1e-9)))

    # Hip tilt angle
    hp_dy  = abs(hip_r["y"] - hip_l["y"])
    hp_dx  = abs(hip_r["x"] - hip_l["x"])
    hp_ang = math.degrees(math.atan2(hp_dy, max(hp_dx, 1e-9)))

    combined = (sh_ang + hp_ang) / 2

    severity = _severity_from_value(combined,
        THRESHOLDS["scoliosis_mild"], THRESHOLDS["scoliosis_moderate"], THRESHOLDS["scoliosis_severe"])

    return {**result, "severity": severity, "angle": round(combined, 1), "unit": "deg",
            "ideal": "< 3°", "measured": f"{combined:.1f}° (sh: {sh_ang:.1f}°, hip: {hp_ang:.1f}°)",
            "landmarks": [LM["l_shoulder"], LM["r_shoulder"], LM["l_hip"], LM["r_hip"]]}


def _check_knee_alignment(lms: list[dict]) -> dict:
    """Genu Valgum / Varum — knee gap vs ankle gap ratio."""
    result = {"condition": "Knee Alignment", "code": "KNEE_ALIGNMENT"}

    kn_l,  kn_r  = _pt(lms, "l_knee"),  _pt(lms, "r_knee")
    ank_l, ank_r = _pt(lms, "l_ankle"), _pt(lms, "r_ankle")

    if not (kn_l and kn_r and ank_l and ank_r):
        return {**result, "severity": "Unknown", "angle": None, "note": "Knee/ankle landmarks not visible"}

    knee_gap  = abs(kn_r["x"]  - kn_l["x"])
    ankle_gap = abs(ank_r["x"] - ank_l["x"])

    if ankle_gap < 1e-4:
        return {**result, "severity": "Unknown", "angle": None, "note": "Ankles too close to measure"}

    ratio = knee_gap / ankle_gap

    # Classify: Ratio < 1 = knock knees (valgum), ratio > 1 = bow legs (varum)
    if ratio < THRESHOLDS["valgum_severe"]:
        severity, kind = "Severe", "Genu Valgum (Knock-Knee)"
    elif ratio < THRESHOLDS["valgum_moderate"]:
        severity, kind = "Moderate", "Genu Valgum (Knock-Knee)"
    elif ratio < THRESHOLDS["valgum_mild"]:
        severity, kind = "Mild", "Genu Valgum (Knock-Knee)"
    elif ratio > THRESHOLDS["varum_severe"]:
        severity, kind = "Severe", "Genu Varum (Bow-Leg)"
    elif ratio > THRESHOLDS["varum_moderate"]:
        severity, kind = "Moderate", "Genu Varum (Bow-Leg)"
    elif ratio > THRESHOLDS["varum_mild"]:
        severity, kind = "Mild", "Genu Varum (Bow-Leg)"
    else:
        severity, kind = "Normal", "Normal"

    return {**result, "severity": severity, "angle": round(ratio, 3), "unit": "ratio",
            "ideal": "0.8 – 1.25", "measured": f"{ratio:.3f} ({kind})",
            "landmarks": [LM["l_knee"], LM["r_knee"], LM["l_ankle"], LM["r_ankle"]]}


# ─── Main entry point ─────────────────────────────────────────────────────────

def classify_posture_deformity(landmarks: list[dict], debug: bool = False) -> dict:
    """
    Run all 6 postural condition checks on a set of MediaPipe landmarks.

    Parameters
    ----------
    landmarks : list of {"x": float, "y": float, "z": float, "visibility": float}
                All values normalised to [0, 1] relative to image dimensions.
    debug     : If True, include raw angle data in output.

    Returns
    -------
    dict with:
        conditions   : list of condition result dicts
        overall      : "Normal" | "Abnormal"
        worst         : name of the most severe condition
        summary_label : short human-readable summary for the main UI card
        debug_info   : raw angles if debug=True
    """
    checks = [
        _check_forward_head,
        _check_kyphosis,
        _check_lordosis,
        _check_pelvic_tilt,
        _check_scoliosis,
        _check_knee_alignment,
    ]

    conditions = []
    severity_order = {"Normal": 0, "Unknown": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

    for fn in checks:
        try:
            cond = fn(landmarks)
            conditions.append(cond)
        except Exception as e:
            logger.warning(f"Condition check {fn.__name__} failed: {e}")

    # Pick worst
    worst_cond = max(conditions, key=lambda c: severity_order.get(c.get("severity", "Normal"), 0))
    worst_severity = worst_cond.get("severity", "Normal")
    
    abnormal_count = sum(1 for c in conditions if severity_order.get(c.get("severity", "Normal"), 0) >= 1)
    overall = "Abnormal" if abnormal_count > 0 else "Normal"

    # Build summary label
    abnormal_names = [c["condition"] for c in conditions if severity_order.get(c.get("severity", "Normal"), 0) >= 1]
    if not abnormal_names:
        summary_label = "Good Posture"
    elif len(abnormal_names) == 1:
        summary_label = abnormal_names[0]
    else:
        summary_label = f"{abnormal_names[0]} (+{len(abnormal_names)-1} more)"

    result = {
        "overall": overall,
        "summary_label": summary_label,
        "worst_condition": worst_cond.get("condition", "Normal"),
        "worst_severity": worst_severity,
        "abnormal_count": abnormal_count,
        "conditions": conditions,
    }

    if debug:
        result["debug_info"] = {
            "raw_conditions": conditions,
            "thresholds_used": THRESHOLDS,
        }

    return result
