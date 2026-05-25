"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        AI DRIVER MONITORING SYSTEM — ML Training Pipeline  v1.0            ║
║                                                                              ║
║  Trains a fatigue/drowsiness classifier that mirrors driver_monitor.py v2.0 ║
║                                                                              ║
║  Features used (matching driver_monitor.py exactly):                        ║
║    • EAR  (Eye Aspect Ratio)      — LEFT_EAR_PTS / RIGHT_EAR_PTS           ║
║    • MAR  (Mouth Aspect Ratio)    — MOUTH_PTS                               ║
║    • PERCLOS  (% eye closure)     — PERCLOS_WINDOW_S = 10s                 ║
║    • Blink rate / slow-blink ratio / mean blink duration                    ║
║    • Head pitch / yaw angles      — solvePnP  (POSE_LANDMARK_IDS)          ║
║    • Microsleep duration (s)                                                 ║
║    • Yawn count / yawns-per-minute                                           ║
║                                                                              ║
║  Targets (FatigueLevel):                                                     ║
║    0 = GREEN   (score  0–29)                                                 ║
║    1 = YELLOW  (score 30–49)                                                 ║
║    2 = ORANGE  (score 50–74)                                                 ║
║    3 = RED     (score 75–100)                                                ║
║                                                                              ║
║  Stack: scikit-learn · XGBoost · PyTorch (optional) · pandas · numpy       ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    # 1. Collect data from driver_monitor.py sessions (CSV in sessions/)
    python train.py --mode collect   # runs live collection via webcam
    python train.py --mode train     # trains on collected CSVs
    python train.py --mode evaluate  # evaluates saved model on test split
    python train.py --mode full      # collect + train + evaluate in sequence
    python train.py --mode predict   # real-time prediction demo
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import csv
import time
import math
import wave
import struct
import signal
import sqlite3
import logging
import argparse
import threading
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

warnings.filterwarnings("ignore", category=UserWarning)

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("[FATAL] opencv-python not installed.\n  Run: pip install opencv-python")

try:
    import numpy as np
except ImportError:
    sys.exit("[FATAL] numpy not installed.\n  Run: pip install numpy")

try:
    from scipy.spatial import distance as dist
except ImportError:
    sys.exit("[FATAL] scipy not installed.\n  Run: pip install scipy")

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise ImportError("Wrong mediapipe version — need 0.10.14")
except ImportError:
    sys.exit("[FATAL] mediapipe not installed correctly.\n"
             "  Run: pip install mediapipe==0.10.14")

try:
    import pandas as pd
except ImportError:
    sys.exit("[FATAL] pandas not installed.\n  Run: pip install pandas")

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score, roc_auc_score)
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import permutation_importance
    import joblib
except ImportError:
    sys.exit("[FATAL] scikit-learn not installed.\n  Run: pip install scikit-learn")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed — XGBoost model will be skipped.\n"
          "  Optional: pip install xgboost")

try:
    import matplotlib
    matplotlib.use("Agg")   # headless-safe
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    print("[WARN] matplotlib/seaborn not installed — plots disabled.\n"
          "  Optional: pip install matplotlib seaborn")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrainPipeline")

# ── MediaPipe aliases ─────────────────────────────────────────────────────────
mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═════════════════════════════════════════════════════════════════════════════
#  SETTINGS  — must match driver_monitor.py v2.0 exactly
# ═════════════════════════════════════════════════════════════════════════════

# Camera / frame
CAMERA_INDEX      = 0
FRAME_W           = 640
FRAME_H           = 480

# EAR thresholds (same as driver_monitor.py)
EAR_THRESHOLD     = 0.22
MICROSLEEP_S      = 2.5
BLINK_MIN_S       = 0.05
BLINK_MAX_S       = 0.40
SLOW_BLINK_S      = 0.20
BLINK_RATE_WINDOW = 60

# PERCLOS (same window as driver_monitor.py)
PERCLOS_WINDOW_S  = 10
PERCLOS_ALERT_PCT = 35.0

# MAR / Yawning
MAR_THRESHOLD     = 0.65
YAWN_MIN_S        = 1.0
YAWN_WINDOW_S     = 60
YAWN_RATE_ALERT   = 3

# Head pose
HEAD_PITCH_THRESH = 20
HEAD_YAW_THRESH   = 35
HEAD_POSE_MIN_S   = 2.0

# Fatigue scoring weights (identical to driver_monitor.py)
W_PERCLOS         = 0.30
W_EAR_DEVIATION   = 0.20
W_BLINK_RATE      = 0.15
W_SLOW_BLINK      = 0.10
W_YAWN            = 0.15
W_HEAD_POSE       = 0.10

# Landmark index sets (identical to driver_monitor.py)
LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]
MOUTH_PTS     = [61, 291, 39, 181, 0, 17, 269, 405]

# 3-D model for head pose (identical to driver_monitor.py)
MODEL_3D_PTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,  -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0,  -150.0, -125.0],
], dtype=np.float64)
POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Training config
DATA_DIR          = "sessions"          # CSVs written by driver_monitor.py
COLLECT_DIR       = "collected_data"    # CSVs written by this script
MODEL_DIR         = "models"
FEATURE_CSV       = os.path.join(COLLECT_DIR, "features.csv")
COLLECTION_FPS    = 10                  # frames per second to collect
COLLECT_DURATION  = None               # None = collect until Q pressed

# FatigueLevel labels (same as driver_monitor.py)
LABEL_GREEN  = 0
LABEL_YELLOW = 1
LABEL_ORANGE = 2
LABEL_RED    = 3

LABEL_NAMES = {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"}
LABEL_FROM_STR = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE NAMES  (must stay in sync with extract_features())
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Raw EAR / MAR
    "ear",
    "mar",
    "ear_left",
    "ear_right",

    # PERCLOS
    "perclos",

    # EAR deviation from calibrated threshold
    "ear_deviation",         # threshold - ear  (>0 = eyes more closed)
    "ear_deviation_norm",    # deviation / threshold

    # Blink analytics
    "blink_rate",            # blinks/min over 60-s window
    "slow_blink_ratio",      # fraction that are slow (>= SLOW_BLINK_S)
    "mean_blink_duration_ms",

    # Microsleep
    "microsleep_s",          # current continuous eye-closure seconds

    # Yawn
    "yawns_per_minute",
    "is_yawning",            # 0/1 boolean

    # Head pose
    "pitch",                 # positive = nodding down
    "yaw",                   # positive = turned right
    "roll",
    "abs_yaw",               # |yaw| — symmetry-invariant
    "pitch_norm",            # pitch / HEAD_PITCH_THRESH
    "yaw_norm",              # |yaw|  / HEAD_YAW_THRESH

    # Composite sub-scores (same math as FatigueEngine)
    "sub_perclos",
    "sub_ear",
    "sub_blink",
    "sub_slow_blink",
    "sub_yawn",
    "sub_pose",
    "fatigue_score",         # weighted sum (ground-truth proxy for regression)
]


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS  (verbatim from driver_monitor.py)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ear(landmarks, pts, w, h) -> float:
    """EAR ~0.30 when open, ~0.10 when closed."""
    p = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in pts])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_mar(landmarks, pts, w, h) -> float:
    """
    Mouth Aspect Ratio.
    Uses 8 mouth boundary points.
    MAR ~0.3–0.5 resting, >0.65 = yawn.
    """
    p = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in pts])
    A = dist.euclidean(p[1], p[7])
    B = dist.euclidean(p[2], p[5])
    C = dist.euclidean(p[3], p[4])
    D = dist.euclidean(p[0], p[6])
    return (A + B + C) / (3.0 * D + 1e-6)


def estimate_head_pose(landmarks, w, h) -> Tuple[float, float, float]:
    """
    Estimate pitch, yaw, roll using cv2.solvePnP.
    Positive pitch = head tilted down. Positive yaw = head turned right.
    """
    img_pts = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in POSE_LANDMARK_IDS
    ], dtype=np.float64)
    focal = w
    cam_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(
        MODEL_3D_PTS, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE low-light enhancement (same as driver_monitor.py)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ═════════════════════════════════════════════════════════════════════════════
#  STATEFUL SIGNAL PROCESSORS  (mirror driver_monitor.py classes)
# ═════════════════════════════════════════════════════════════════════════════

class Perclos:
    """Rolling-window PERCLOS (identical to driver_monitor.py)."""

    def __init__(self, window_s: float = PERCLOS_WINDOW_S):
        self._window_s = window_s
        self._buf: deque = deque()

    def update(self, is_closed: bool, now: float) -> float:
        self._buf.append((now, is_closed))
        cutoff = now - self._window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        if not self._buf:
            return 0.0
        n_closed = sum(1 for _, c in self._buf if c)
        return 100.0 * n_closed / len(self._buf)


class BlinkAnalyzer:
    """Blink analytics (identical to driver_monitor.py)."""

    def __init__(self, rate_window_s: float = BLINK_RATE_WINDOW):
        self._window     = rate_window_s
        self._blink_log: deque = deque()
        self._in_blink   = False
        self._blink_start: Optional[float] = None
        self.total_blinks  = 0
        self.slow_blinks   = 0
        self.last_duration = 0.0

    def update(self, eyes_closed: bool, now: float) -> None:
        if eyes_closed and not self._in_blink:
            self._in_blink    = True
            self._blink_start = now
        elif not eyes_closed and self._in_blink:
            self._in_blink = False
            dur = now - (self._blink_start or now)
            if BLINK_MIN_S <= dur <= BLINK_MAX_S:
                self.total_blinks  += 1
                self.last_duration  = dur
                is_slow = dur >= SLOW_BLINK_S
                if is_slow:
                    self.slow_blinks += 1
                self._blink_log.append((now, dur, is_slow))
        cutoff = now - self._window
        while self._blink_log and self._blink_log[0][0] < cutoff:
            self._blink_log.popleft()

    @property
    def blink_rate(self) -> float:
        if not self._blink_log:
            return 0.0
        return len(self._blink_log) / (self._window / 60.0)

    @property
    def slow_blink_ratio(self) -> float:
        if not self._blink_log:
            return 0.0
        slow = sum(1 for _, _, s in self._blink_log if s)
        return slow / len(self._blink_log)

    @property
    def mean_duration_ms(self) -> float:
        if not self._blink_log:
            return 0.0
        return 1000.0 * float(np.mean([d for _, d, _ in self._blink_log]))

    def blink_fatigue_score(self) -> float:
        """0–100 sub-score (identical to driver_monitor.py)."""
        rate = self.blink_rate
        if rate < 8:
            rate_score = 100.0
        elif rate > 20:
            rate_score = 0.0
        else:
            rate_score = max(0.0, (20.0 - rate) / 12.0 * 80.0)
        slow_score = self.slow_blink_ratio * 100.0
        return min(100.0, rate_score * 0.5 + slow_score * 0.5)


class YawnTracker:
    """Yawn tracker (identical to driver_monitor.py)."""

    def __init__(self):
        self._in_yawn    = False
        self._yawn_start: Optional[float] = None
        self._yawn_log: deque = deque()
        self.total_yawns = 0
        self.is_yawning  = False

    def update(self, mar: float, threshold: float, now: float) -> None:
        above = mar > threshold
        if above and not self._in_yawn:
            self._in_yawn    = True
            self._yawn_start = now
        elif not above and self._in_yawn:
            self._in_yawn = False
            dur = now - (self._yawn_start or now)
            if dur >= YAWN_MIN_S:
                self.total_yawns += 1
                self._yawn_log.append(now)
        self.is_yawning = self._in_yawn and (
            now - (self._yawn_start or now) >= 0.5
        )
        cutoff = now - YAWN_WINDOW_S
        while self._yawn_log and self._yawn_log[0] < cutoff:
            self._yawn_log.popleft()

    @property
    def yawns_per_minute(self) -> float:
        return len(self._yawn_log) / (YAWN_WINDOW_S / 60.0)

    def yawn_fatigue_score(self) -> float:
        """0–100 sub-score (identical to driver_monitor.py)."""
        return min(100.0, (self.yawns_per_minute / YAWN_RATE_ALERT) * 100.0)


class HeadPoseMonitor:
    """Head pose monitor (identical to driver_monitor.py)."""

    def __init__(self):
        self._distracted_start: Optional[float] = None
        self.pitch         = 0.0
        self.yaw           = 0.0
        self.roll          = 0.0
        self.is_distracted = False
        self.distraction_type = ""

    def update(self, pitch: float, yaw: float, roll: float, now: float) -> None:
        self.pitch = pitch
        self.yaw   = yaw
        self.roll  = roll
        down      = pitch > HEAD_PITCH_THRESH
        sideways  = abs(yaw) > HEAD_YAW_THRESH
        distracted = down or sideways
        if distracted:
            if self._distracted_start is None:
                self._distracted_start = now
            elapsed = now - self._distracted_start
            self.is_distracted = elapsed >= HEAD_POSE_MIN_S
            self.distraction_type = "HEAD DOWN" if down else "LOOKING AWAY"
        else:
            self._distracted_start = None
            self.is_distracted     = False
            self.distraction_type  = ""

    def pose_fatigue_score(self) -> float:
        """0–100 sub-score (identical to driver_monitor.py)."""
        p = min(100.0, max(0.0, (self.pitch - 5.0) / HEAD_PITCH_THRESH * 100.0))
        y = min(100.0, max(0.0, (abs(self.yaw) - 5.0) / HEAD_YAW_THRESH * 100.0))
        return max(p, y)


class Calibrator:
    """
    Per-session EAR/MAR calibration (identical to driver_monitor.py).
    Collects first N frames with open eyes, derives personalized threshold.
    """

    N_FRAMES   = 90          # ~3 seconds at 30 fps
    EAR_MARGIN = 0.08        # threshold = baseline_ear - margin
    MAR_MARGIN = 0.20

    def __init__(self):
        self._ear_buf: List[float] = []
        self._mar_buf: List[float] = []
        self.done      = False
        self.threshold = EAR_THRESHOLD
        self.mar_threshold = MAR_THRESHOLD

    def feed(self, ear: float, mar: float) -> None:
        if self.done:
            return
        self._ear_buf.append(ear)
        self._mar_buf.append(mar)
        if len(self._ear_buf) >= self.N_FRAMES:
            baseline_ear = float(np.percentile(self._ear_buf, 70))
            self.threshold     = max(0.10, baseline_ear - self.EAR_MARGIN)
            baseline_mar = float(np.percentile(self._mar_buf, 80))
            self.mar_threshold = max(0.40, baseline_mar + self.MAR_MARGIN)
            self.done = True
            log.info(f"Calibration done  EAR_thr={self.threshold:.3f}  MAR_thr={self.mar_threshold:.3f}")

    @property
    def progress(self) -> float:
        return min(1.0, len(self._ear_buf) / self.N_FRAMES)


# ═════════════════════════════════════════════════════════════════════════════
#  FATIGUE SCORE COMPUTATION  (identical formula to FatigueEngine)
# ═════════════════════════════════════════════════════════════════════════════

def compute_fatigue_score(
    perclos_pct: float,
    ear_val: float,
    ear_threshold: float,
    blink: BlinkAnalyzer,
    yawn: YawnTracker,
    head: HeadPoseMonitor,
    microsleep_s: float = 0.0,
) -> Tuple[float, int]:
    """
    Compute fatigue score 0–100 and label 0–3.
    Identical formula to driver_monitor.py FatigueEngine.update().
    Returns (score, label).
    """
    perclos_sub = min(100.0, perclos_pct / PERCLOS_ALERT_PCT * 100.0)

    if ear_threshold > 0:
        ear_sub = max(0.0, (ear_threshold - ear_val) / ear_threshold * 200.0)
        ear_sub = min(100.0, ear_sub)
    else:
        ear_sub = 0.0

    blink_sub = blink.blink_fatigue_score()
    slow_sub  = blink.slow_blink_ratio * 100.0
    yawn_sub  = yawn.yawn_fatigue_score()
    pose_sub  = head.pose_fatigue_score()

    raw = (
        W_PERCLOS       * perclos_sub +
        W_EAR_DEVIATION * ear_sub     +
        W_BLINK_RATE    * blink_sub   +
        W_SLOW_BLINK    * slow_sub    +
        W_YAWN          * yawn_sub    +
        W_HEAD_POSE     * pose_sub
    )

    if microsleep_s >= MICROSLEEP_S:
        bonus = min(50.0, (microsleep_s - MICROSLEEP_S) * 15.0 + 30.0)
        raw   = min(100.0, raw + bonus)

    # Fatigue level label (matches FatigueLevel.from_score)
    if raw < 30:
        label = LABEL_GREEN
    elif raw < 50:
        label = LABEL_YELLOW
    elif raw < 75:
        label = LABEL_ORANGE
    else:
        label = LABEL_RED

    return raw, label


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  — produces one row of FEATURE_NAMES per frame
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(
    ear_val: float,
    ear_left: float,
    ear_right: float,
    mar_val: float,
    perclos_val: float,
    threshold: float,
    blink: BlinkAnalyzer,
    yawn: YawnTracker,
    head: HeadPoseMonitor,
    microsleep_s: float,
) -> Dict[str, float]:
    """
    Build the feature dict for a single frame.
    Sub-scores are recomputed here so they are available as individual features
    for ML models that can use them directly.
    """
    perclos_sub = min(100.0, perclos_val / PERCLOS_ALERT_PCT * 100.0)

    if threshold > 0:
        ear_sub = max(0.0, (threshold - ear_val) / threshold * 200.0)
        ear_sub = min(100.0, ear_sub)
    else:
        ear_sub = 0.0

    blink_sub = blink.blink_fatigue_score()
    slow_sub  = blink.slow_blink_ratio * 100.0
    yawn_sub  = yawn.yawn_fatigue_score()
    pose_sub  = head.pose_fatigue_score()

    raw = (
        W_PERCLOS       * perclos_sub +
        W_EAR_DEVIATION * ear_sub     +
        W_BLINK_RATE    * blink_sub   +
        W_SLOW_BLINK    * slow_sub    +
        W_YAWN          * yawn_sub    +
        W_HEAD_POSE     * pose_sub
    )

    if microsleep_s >= MICROSLEEP_S:
        bonus = min(50.0, (microsleep_s - MICROSLEEP_S) * 15.0 + 30.0)
        raw   = min(100.0, raw + bonus)

    return {
        "ear":                  ear_val,
        "mar":                  mar_val,
        "ear_left":             ear_left,
        "ear_right":            ear_right,
        "perclos":              perclos_val,
        "ear_deviation":        max(0.0, threshold - ear_val),
        "ear_deviation_norm":   max(0.0, threshold - ear_val) / (threshold + 1e-6),
        "blink_rate":           blink.blink_rate,
        "slow_blink_ratio":     blink.slow_blink_ratio,
        "mean_blink_duration_ms": blink.mean_duration_ms,
        "microsleep_s":         microsleep_s,
        "yawns_per_minute":     yawn.yawns_per_minute,
        "is_yawning":           float(yawn.is_yawning),
        "pitch":                head.pitch,
        "yaw":                  head.yaw,
        "roll":                 head.roll,
        "abs_yaw":              abs(head.yaw),
        "pitch_norm":           head.pitch / (HEAD_PITCH_THRESH + 1e-6),
        "yaw_norm":             abs(head.yaw) / (HEAD_YAW_THRESH + 1e-6),
        "sub_perclos":          perclos_sub,
        "sub_ear":              ear_sub,
        "sub_blink":            blink_sub,
        "sub_slow_blink":       slow_sub,
        "sub_yawn":             yawn_sub,
        "sub_pose":             pose_sub,
        "fatigue_score":        raw,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION  — live webcam → feature CSV
# ═════════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    Captures live webcam frames using MediaPipe face mesh,
    extracts all features, prompts the user to label each window,
    and appends rows to FEATURE_CSV.

    Label entry:
      Press 0 = GREEN / alert
      Press 1 = YELLOW / mild fatigue
      Press 2 = ORANGE / moderate fatigue
      Press 3 = RED    / severe fatigue / microsleep
      Q / ESC = quit collection
    """

    def __init__(self, camera_index: int = CAMERA_INDEX,
                 output_csv: str = FEATURE_CSV,
                 use_clahe: bool = False):
        self._cam_idx   = camera_index
        self._out_csv   = output_csv
        self._clahe     = use_clahe
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

        self._write_header = not os.path.exists(output_csv)
        self._csv_file = open(output_csv, "a", newline="")
        self._writer   = csv.DictWriter(
            self._csv_file, fieldnames=["timestamp", "label"] + FEATURE_NAMES
        )
        if self._write_header:
            self._writer.writeheader()
            log.info(f"Created {output_csv}")
        else:
            log.info(f"Appending to {output_csv}")

        # Signal processors (mirrors driver_monitor.py DriverMonitor.__init__)
        self.calib       = Calibrator()
        self.perclos_buf = Perclos()
        self.blink       = BlinkAnalyzer()
        self.yawn        = YawnTracker()
        self.head        = HeadPoseMonitor()

        self.eye_closed_start: Optional[float] = None
        self.microsleep_s = 0.0
        self._current_label = LABEL_GREEN
        self._rows_written  = 0

    def run(self) -> None:
        cap = cv2.VideoCapture(self._cam_idx)
        if not cap.isOpened():
            log.error(f"Cannot open camera {self._cam_idx}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        log.info("Data collection started.")
        log.info("  Keys: 0=GREEN  1=YELLOW  2=ORANGE  3=RED  Q=quit")

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            frame_delay = int(1000 / COLLECTION_FPS)

            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                now   = time.monotonic()
                h, w  = frame.shape[:2]
                frame = cv2.flip(frame, 1)

                if self._clahe:
                    frame = apply_clahe(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                ear_val = mar_val = ear_left = ear_right = 0.0
                pitch = yaw = roll = 0.0

                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark

                    ear_left  = compute_ear(lms, LEFT_EAR_PTS,  w, h)
                    ear_right = compute_ear(lms, RIGHT_EAR_PTS, w, h)
                    ear_val   = (ear_left + ear_right) / 2.0
                    mar_val   = compute_mar(lms, MOUTH_PTS, w, h)

                    try:
                        pitch, yaw, roll = estimate_head_pose(lms, w, h)
                    except Exception:
                        pass

                    self.calib.feed(ear_val, mar_val)
                    threshold     = self.calib.threshold
                    mar_threshold = self.calib.mar_threshold

                    is_closed = ear_val < threshold
                    self.blink.update(is_closed, now)
                    self.yawn.update(mar_val, mar_threshold, now)
                    self.head.update(pitch, yaw, roll, now)
                    perclos_val = self.perclos_buf.update(is_closed, now)

                    if is_closed:
                        if self.eye_closed_start is None:
                            self.eye_closed_start = now
                        self.microsleep_s = now - self.eye_closed_start
                    else:
                        self.eye_closed_start = None
                        self.microsleep_s     = 0.0

                    feat = extract_features(
                        ear_val, ear_left, ear_right, mar_val, perclos_val,
                        threshold, self.blink, self.yawn, self.head,
                        self.microsleep_s
                    )

                    row = {"timestamp": datetime.now().isoformat(timespec="milliseconds"),
                           "label": self._current_label}
                    row.update(feat)
                    self._writer.writerow(row)
                    self._rows_written += 1

                # ── HUD overlay for collection ────────────────────────────────
                lv_color_map = {0: (50,200,50), 1:(0,210,220), 2:(0,145,255), 3:(30,30,220)}
                color = lv_color_map[self._current_label]
                cv2.putText(frame,
                    f"Label: {LABEL_NAMES[self._current_label]}  |  "
                    f"Rows: {self._rows_written}  |  "
                    f"EAR: {ear_val:.3f}  MAR: {mar_val:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
                cv2.putText(frame,
                    f"Label: {LABEL_NAMES[self._current_label]}  |  "
                    f"Rows: {self._rows_written}  |  "
                    f"EAR: {ear_val:.3f}  MAR: {mar_val:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame,
                    "Keys: 0=GREEN 1=YELLOW 2=ORANGE 3=RED  Q=Quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                if not self.calib.done:
                    prog = int(self.calib.progress * w)
                    cv2.rectangle(frame, (0, h-6), (prog, h), (0,210,210), -1)
                    cv2.putText(frame,
                        f"Calibrating {self.calib.progress*100:.0f}% ...",
                        (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,210,210), 1)

                cv2.imshow("Data Collection", frame)
                key = cv2.waitKey(frame_delay) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                    self._current_label = int(chr(key))
                    log.info(f"Label → {LABEL_NAMES[self._current_label]}")

        cap.release()
        cv2.destroyAllWindows()
        self._csv_file.flush()
        self._csv_file.close()
        log.info(f"Collection done — {self._rows_written} rows written to {self._out_csv}")


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING  — merges driver_monitor.py session CSVs + collected CSVs
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(
    session_dir: str = DATA_DIR,
    collect_dir: str = COLLECT_DIR,
) -> pd.DataFrame:
    """
    Load and merge:
      1. Session CSVs from driver_monitor.py  (columns: level, ear, mar, …)
      2. Feature CSVs from DataCollector       (columns: label, + FEATURE_NAMES)

    Returns a combined DataFrame with columns FEATURE_NAMES + "label".
    """
    frames = []

    # ── 1. driver_monitor.py session CSVs ────────────────────────────────────
    session_csvs = list(Path(session_dir).glob("session_*.csv")) if os.path.isdir(session_dir) else []
    for path in session_csvs:
        try:
            df = pd.read_csv(path)
            # Map string level → numeric label
            if "level" in df.columns:
                df["label"] = df["level"].map(LABEL_FROM_STR)
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)
            # Only keep columns present in FEATURE_NAMES
            available = [c for c in FEATURE_NAMES if c in df.columns]
            df = df[available + ["label"]].copy()
            # Fill missing features with 0
            for col in FEATURE_NAMES:
                if col not in df.columns:
                    df[col] = 0.0
            frames.append(df[FEATURE_NAMES + ["label"]])
            log.info(f"  Loaded session CSV: {path.name}  ({len(df)} rows)")
        except Exception as exc:
            log.warning(f"  Could not load {path}: {exc}")

    # ── 2. Feature CSVs from DataCollector ───────────────────────────────────
    collect_csvs = list(Path(collect_dir).glob("*.csv")) if os.path.isdir(collect_dir) else []
    for path in collect_csvs:
        try:
            df = pd.read_csv(path)
            if "label" not in df.columns:
                log.warning(f"  Skipping {path.name} — no 'label' column")
                continue
            for col in FEATURE_NAMES:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[FEATURE_NAMES + ["label"]].copy()
            df["label"] = df["label"].astype(int)
            frames.append(df)
            log.info(f"  Loaded collector CSV: {path.name}  ({len(df)} rows)")
        except Exception as exc:
            log.warning(f"  Could not load {path}: {exc}")

    if not frames:
        log.error("No data found! Run with --mode collect first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna()
    log.info(f"Dataset: {len(combined)} rows, "
             f"labels={combined['label'].value_counts().to_dict()}")
    return combined


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def train(data_dir: str = DATA_DIR, collect_dir: str = COLLECT_DIR,
          model_dir: str = MODEL_DIR) -> None:
    """
    Train multiple classifiers on the collected feature data.
    Saves the best model + scaler to MODEL_DIR.
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_dataset(data_dir, collect_dir)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    y  = df["label"].values.astype(int)

    # ── Train / test split (stratified) ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model zoo ─────────────────────────────────────────────────────────────
    models: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    }

    if _HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
        )

    best_name  = None
    best_score = -1.0
    results    = {}

    log.info("═" * 60)
    log.info("  Training models …")
    log.info("═" * 60)

    for name, model in models.items():
        log.info(f"\n── {name} ──────────────────────────────────────────")

        X_fit = X_train_s if name == "LogisticRegression" else X_train
        X_eval = X_test_s  if name == "LogisticRegression" else X_test

        model.fit(X_fit, y_train)
        preds  = model.predict(X_eval)
        acc    = accuracy_score(y_test, preds)
        f1     = f1_score(y_test, preds, average="weighted", zero_division=0)

        log.info(f"  Accuracy : {acc:.4f}")
        log.info(f"  F1 (weighted): {f1:.4f}")
        log.info("\n" + classification_report(
            y_test, preds,
            target_names=[LABEL_NAMES[i] for i in range(4)],
            zero_division=0,
        ))

        results[name] = {"accuracy": acc, "f1": f1, "model": model,
                         "preds": preds, "X_eval": X_eval}

        if f1 > best_score:
            best_score = f1
            best_name  = name

    log.info("═" * 60)
    log.info(f"  Best model: {best_name}  (F1={best_score:.4f})")
    log.info("═" * 60)

    # ── Save best model + scaler ──────────────────────────────────────────────
    best_model = results[best_name]["model"]
    model_path  = os.path.join(model_dir, "best_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path   = os.path.join(model_dir, "meta.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "best_model_name": best_name,
        "feature_names":   FEATURE_NAMES,
        "label_names":     LABEL_NAMES,
        "trained_at":      datetime.now().isoformat(),
        "train_rows":      len(X_train),
        "test_f1":         best_score,
    }, meta_path)

    log.info(f"  Saved: {model_path}")
    log.info(f"  Saved: {scaler_path}")
    log.info(f"  Saved: {meta_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if _HAS_PLOT:
        _plot_results(results, best_name, y_test, model_dir)

    return best_model, scaler


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model_dir: str = MODEL_DIR,
             data_dir:  str = DATA_DIR,
             collect_dir: str = COLLECT_DIR) -> None:
    """Load saved model and evaluate on a fresh test split."""
    model_path  = os.path.join(model_dir, "best_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path   = os.path.join(model_dir, "meta.pkl")

    for p in (model_path, scaler_path, meta_path):
        if not os.path.exists(p):
            log.error(f"File not found: {p}  — run --mode train first")
            return

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta   = joblib.load(meta_path)

    log.info(f"Loaded model: {meta['best_model_name']}  "
             f"(trained {meta['trained_at']},  F1={meta['test_f1']:.4f})")

    df = load_dataset(data_dir, collect_dir)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    y  = df["label"].values.astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test_s = scaler.transform(X_test)

    # Use scaled data for LogReg, raw for tree-based
    needs_scale = "Logistic" in meta["best_model_name"]
    X_eval = X_test_s if needs_scale else X_test
    preds  = model.predict(X_eval)

    log.info("\n" + "=" * 60)
    log.info("EVALUATION REPORT")
    log.info("=" * 60)
    log.info(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    log.info(f"F1 (weighted): {f1_score(y_test, preds, average='weighted', zero_division=0):.4f}")
    log.info("\n" + classification_report(
        y_test, preds,
        target_names=[LABEL_NAMES[i] for i in range(4)],
        zero_division=0,
    ))

    if _HAS_PLOT:
        _plot_confusion(y_test, preds, model_dir, prefix="eval")

    # ── Feature importances ───────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        top = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:10]
        log.info("\nTop-10 feature importances:")
        for fname, score in top:
            log.info(f"  {fname:<28s}  {score:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  REAL-TIME PREDICTION  — loads saved model and runs live inference
# ═════════════════════════════════════════════════════════════════════════════

class LivePredictor:
    """
    Loads the saved model and runs live inference on webcam frames.
    Overlays the predicted FatigueLevel + confidence on screen.
    All signal processing mirrors driver_monitor.py exactly.
    """

    def __init__(self, model_dir: str = MODEL_DIR,
                 camera_index: int = CAMERA_INDEX,
                 use_clahe: bool = False):
        model_path  = os.path.join(model_dir, "best_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        meta_path   = os.path.join(model_dir, "meta.pkl")
        for p in (model_path, scaler_path, meta_path):
            if not os.path.exists(p):
                log.error(f"Missing {p} — run --mode train first")
                sys.exit(1)

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.meta   = joblib.load(meta_path)
        self._needs_scale = "Logistic" in self.meta["best_model_name"]
        self._cam_idx = camera_index
        self._clahe   = use_clahe

        # Signal processors (mirrors DriverMonitor.__init__)
        self.calib    = Calibrator()
        self.perclos  = Perclos()
        self.blink    = BlinkAnalyzer()
        self.yawn     = YawnTracker()
        self.head     = HeadPoseMonitor()
        self.eye_closed_start: Optional[float] = None
        self.microsleep_s = 0.0

        log.info(f"LivePredictor ready — model: {self.meta['best_model_name']}")

    def _predict(self, feat: Dict[str, float]) -> Tuple[int, float]:
        x = np.array([[feat[k] for k in FEATURE_NAMES]], dtype=np.float32)
        if self._needs_scale:
            x = self.scaler.transform(x)
        label = int(self.model.predict(x)[0])
        probs = self.model.predict_proba(x)[0] if hasattr(self.model, "predict_proba") else None
        conf  = float(probs[label]) if probs is not None else 1.0
        return label, conf

    def run(self) -> None:
        cap = cv2.VideoCapture(self._cam_idx)
        if not cap.isOpened():
            log.error(f"Cannot open camera {self._cam_idx}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        log.info("Live prediction started — press Q to quit")

        color_map = {0:(50,200,50), 1:(0,210,220), 2:(0,145,255), 3:(30,30,220)}

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                now   = time.monotonic()
                h, w  = frame.shape[:2]
                frame = cv2.flip(frame, 1)
                if self._clahe:
                    frame = apply_clahe(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                ear_val = mar_val = ear_left = ear_right = 0.0
                label, conf = 0, 0.0

                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark

                    ear_left  = compute_ear(lms, LEFT_EAR_PTS,  w, h)
                    ear_right = compute_ear(lms, RIGHT_EAR_PTS, w, h)
                    ear_val   = (ear_left + ear_right) / 2.0
                    mar_val   = compute_mar(lms, MOUTH_PTS, w, h)

                    pitch = yaw = roll = 0.0
                    try:
                        pitch, yaw, roll = estimate_head_pose(lms, w, h)
                    except Exception:
                        pass

                    self.calib.feed(ear_val, mar_val)
                    threshold     = self.calib.threshold
                    mar_threshold = self.calib.mar_threshold

                    is_closed = ear_val < threshold
                    self.blink.update(is_closed, now)
                    self.yawn.update(mar_val, mar_threshold, now)
                    self.head.update(pitch, yaw, roll, now)
                    perclos_val = self.perclos.update(is_closed, now)

                    if is_closed:
                        if self.eye_closed_start is None:
                            self.eye_closed_start = now
                        self.microsleep_s = now - self.eye_closed_start
                    else:
                        self.eye_closed_start = None
                        self.microsleep_s     = 0.0

                    feat  = extract_features(
                        ear_val, ear_left, ear_right, mar_val, perclos_val,
                        threshold, self.blink, self.yawn, self.head,
                        self.microsleep_s
                    )
                    label, conf = self._predict(feat)

                # ── Overlay ───────────────────────────────────────────────────
                color = color_map[label]
                cv2.rectangle(frame, (0, 0), (w, 50), (15,15,15), -1)
                cv2.putText(frame,
                    f"ML Prediction: {LABEL_NAMES[label]}  ({conf*100:.0f}%)  "
                    f"EAR:{ear_val:.3f}  MAR:{mar_val:.3f}",
                    (10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)

                if not self.calib.done:
                    prog = int(self.calib.progress * w)
                    cv2.rectangle(frame, (0, h-6), (prog, h), (0,210,210), -1)

                cv2.imshow("Live Prediction", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        cap.release()
        cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _plot_confusion(y_true, y_pred, model_dir: str, prefix: str = "train") -> None:
    if not _HAS_PLOT:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[LABEL_NAMES[i] for i in range(4)],
                yticklabels=[LABEL_NAMES[i] for i in range(4)], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({prefix})")
    plt.tight_layout()
    out = os.path.join(model_dir, f"{prefix}_confusion.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"  Saved plot: {out}")


def _plot_results(results: dict, best_name: str, y_test, model_dir: str) -> None:
    if not _HAS_PLOT:
        return

    # Accuracy / F1 bar chart
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    f1s    = [results[n]["f1"]       for n in names]
    x      = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, accs, 0.35, label="Accuracy", color="#4CAF50")
    ax.bar(x + 0.18, f1s,  0.35, label="F1 (weighted)", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(model_dir, "model_comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"  Saved plot: {out}")

    # Confusion matrix for best model
    _plot_confusion(y_test, results[best_name]["preds"], model_dir, prefix="best")

    # Feature importance (if available)
    best_model = results[best_name]["model"]
    if hasattr(best_model, "feature_importances_"):
        imp  = best_model.feature_importances_
        idx  = np.argsort(imp)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([FEATURE_NAMES[i] for i in idx[::-1]],
                [imp[i] for i in idx[::-1]], color="#FF5722")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-15 Feature Importances ({best_name})")
        plt.tight_layout()
        out = os.path.join(model_dir, "feature_importance.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log.info(f"  Saved plot: {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AI Driver Monitor — ML Training Pipeline v1.0"
    )
    p.add_argument(
        "--mode", required=True,
        choices=["collect", "train", "evaluate", "full", "predict"],
        help=(
            "collect  = live webcam data collection\n"
            "train    = train models on collected data\n"
            "evaluate = evaluate saved model\n"
            "full     = collect + train + evaluate\n"
            "predict  = live webcam prediction with saved model"
        ),
    )
    p.add_argument("--camera",    type=int, default=CAMERA_INDEX,
                   help="Camera index (default: 0)")
    p.add_argument("--data_dir",  default=DATA_DIR,
                   help=f"Session CSV directory (default: {DATA_DIR})")
    p.add_argument("--collect_dir", default=COLLECT_DIR,
                   help=f"Collected features directory (default: {COLLECT_DIR})")
    p.add_argument("--model_dir", default=MODEL_DIR,
                   help=f"Model save directory (default: {MODEL_DIR})")
    p.add_argument("--clahe",     action="store_true",
                   help="Enable CLAHE low-light enhancement during collection")
    return p.parse_args()


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   AI Driver Monitor — ML Training Pipeline  v1.0            ║
╚══════════════════════════════════════════════════════════════╝
""")

    args = parse_args()

    if args.mode in ("collect", "full"):
        log.info("Starting data collection …")
        collector = DataCollector(
            camera_index=args.camera,
            output_csv=os.path.join(args.collect_dir, "features.csv"),
            use_clahe=args.clahe,
        )
        collector.run()

    if args.mode in ("train", "full"):
        log.info("Starting training …")
        train(
            data_dir=args.data_dir,
            collect_dir=args.collect_dir,
            model_dir=args.model_dir,
        )

    if args.mode in ("evaluate", "full"):
        log.info("Evaluating saved model …")
        evaluate(
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            collect_dir=args.collect_dir,
        )

    if args.mode == "predict":
        log.info("Starting live prediction …")
        predictor = LivePredictor(
            model_dir=args.model_dir,
            camera_index=args.camera,
            use_clahe=args.clahe,
        )
        predictor.run()


if __name__ == "__main__":
    main()
