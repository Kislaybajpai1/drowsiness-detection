# 🚗 AI Driver Drowsiness Monitoring System

A real-time, CPU-only driver fatigue detection system built with **MediaPipe Face Mesh**, **OpenCV**, and **scikit-learn**. It watches the driver through a webcam, computes multiple physiological and behavioural fatigue indicators, and raises escalating audio, voice and visual alerts, with an optional **trained ML classifier** running alongside a rule-based fatigue engine.

> No GPU required · Runs on a standard laptop webcam · $0 hardware cost

---

## 📑 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Versions](#-project-versions)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Keyboard Controls](#-keyboard-controls)
- [Configuration](#-configuration)
- [Fatigue Scoring](#-fatigue-scoring)
- [ML Model Integration](#-ml-model-integration)
- [Logging and Outputs](#-logging-and-outputs)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Eye closure detection (EAR)** | Eye Aspect Ratio computed from 6 landmarks per eye on the MediaPipe 478-point mesh |
| **Per-user auto-calibration** | EAR and MAR thresholds are calibrated automatically from the first ~60 valid frames |
| **PERCLOS** | Percentage of eye closure over a rolling 10-second window (alert at ≥ 35%) |
| **Microsleep detection** | Continuous eye closure for ≥ 2.5 s triggers a critical alert that bypasses the cooldown |
| **Blink analytics** | Blink rate, slow-blink ratio (≥ 200 ms) and mean blink duration |
| **Yawn detection (MAR)** | Mouth Aspect Ratio held above threshold for ≥ 1 s counts as a yawn; yawns per minute are tracked |
| **Head pose estimation** | Pitch / yaw / roll via `cv2.solvePnP`; flags "HEAD DOWN" and "LOOKING AWAY" |
| **No-face / camera-block detection** | Alert if no face is visible for > 5 s |
| **Multi-level fatigue score** | 0–100 score mapped to GREEN → YELLOW → ORANGE → RED |
| **ML prediction (v3)** | Trained RandomForest / XGBoost / Logistic Regression model predicts the fatigue level every frame |
| **Dual-mode HUD** | Shows rule-based and ML predictions side by side with an AGREE / CONFLICT indicator |
| **Low-light enhancement** | CLAHE (LAB colour space) toggle for dark cabins |
| **Audio + voice alerts** | `pygame` alarm sounds and optional `pyttsx3` spoken warnings |
| **Camera selection menu** | Auto-detects cameras at indices 0–9 |
| **CSV + SQLite logging** | Every alert is logged with metrics, snapshot path and ML label/confidence |
| **Graceful fallbacks** | Works without pygame (terminal beep), without pyttsx3 (no voice), and without model files (rule-based only) |

---

## 🧠 How It Works

```
Webcam frame
    │
    ▼
MediaPipe Face Mesh (478 landmarks, refine_landmarks=True)
    │
    ├── EAR (left + right) ───────► Eye state ──► PERCLOS / Blink analytics / Microsleep timer
    ├── MAR ──────────────────────► Yawn tracker
    ├── solvePnP head pose ───────► Pitch / Yaw / Roll ──► Head-pose monitor
    └── Face present? ────────────► No-face detector
                    │
                    ▼
        Rule-based Fatigue Engine (weighted sub-scores, 0–100)
                    │
                    ├──► 26-feature vector ──► ML model ──► Level + confidence
                    │
                    ▼
        Alerts (alarm · voice · snapshot · log) + HUD overlay
```

### Core metrics

- **EAR (Eye Aspect Ratio)**: `(‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)`. Roughly 0.30 when open and 0.10 when closed.
- **MAR (Mouth Aspect Ratio)**: mean of three vertical mouth distances divided by the horizontal width. Resting is about 0.3–0.5; above 0.65 indicates a yawn. The threshold must be held for ≥ 1 s to reduce false positives from talking or laughing.
- **PERCLOS**: share of frames in the last 10 s where the eyes are closed.
- **Head pose**: 2D landmarks (nose tip, chin, eye corners, mouth corners) are mapped to a canonical 3D face model with `solvePnP`, then decomposed into Euler angles.

---

## 🔖 Project Versions

| File | Version | Description |
|---|---|---|
| `driver_monitor_base.py` | v2.0 | Rule-based system. No ML dependency. |
| `driver_monitor_trained.py` | v3.0 | Everything in v2.0, plus a trained ML classifier, dual-mode HUD, and ML columns in the logs. |

---

## 📁 Repository Structure

```
.
├── driver_monitor_base.py       # v2.0 – rule-based monitor
├── driver_monitor_trained.py    # v3.0 – rule-based + ML monitor
├── alarm.wav                    # Standard alert sound
├── alarm_critical.wav           # Critical (microsleep) alert sound
├── models/                      # Created by the training script
│   ├── best_model.pkl           #   Best trained classifier
│   ├── scaler.pkl               #   Feature scaler
│   └── meta.pkl                 #   Model name, test F1, training timestamp
├── sessions/                    # Auto-created: CSV logs + events.db (SQLite)
├── drowsy_snapshots/            # Auto-created: JPEG snapshot for each alert
├── requirements.txt
└── README.md
```

> `models/`, `sessions/` and `drowsy_snapshots/` are generated at runtime or by training. If the alarm `.wav` files are missing, the program generates simple beep tones automatically.

---

## ⚙️ Installation

**Requirements:** Python 3.10+ and a webcam.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install core dependencies
pip install mediapipe==0.10.14 opencv-python scipy numpy pygame pyttsx3

# 4. For the ML version (v3.0) also install
pip install scikit-learn xgboost joblib
```

> ⚠️ **MediaPipe must be version `0.10.14`.** Newer releases removed the `mp.solutions` API this project relies on.

`pygame` and `pyttsx3` are optional. Without them the system falls back to a terminal beep and disables voice alerts.

---

## ▶️ Usage

**Rule-based version (v2.0):**
```bash
python driver_monitor_base.py
```

**ML-enhanced version (v3.0):**
```bash
# Train the model first so that models/best_model.pkl and models/scaler.pkl exist
python train.py --mode train

python driver_monitor_trained.py
```

If the model files are missing, v3.0 still runs and shows **"ML: no model loaded"** on the HUD, using the rule-based engine only.

### First launch

1. If multiple cameras are detected, you are prompted to select one.
2. During **calibration** (the first few seconds), look at the camera with your **eyes open and relaxed**. A progress bar is shown at the bottom of the window.
3. Once calibration finishes, monitoring begins automatically.

---

## ⌨️ Keyboard Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `R` | Dismiss the active alert |
| `F` | Toggle face-mesh overlay |
| `C` | Re-run camera selection |
| `L` | Toggle low-light enhancement (CLAHE) |
| `M` | Toggle ML panel on the HUD *(v3.0 only)* |

---

## 🔧 Configuration

All tunable parameters are constants at the top of each script.

| Parameter | Default | Meaning |
|---|---|---|
| `EAR_THRESHOLD` | 0.22 | Fallback eye-closure threshold (overridden by calibration) |
| `MICROSLEEP_S` | 2.5 | Seconds of continuous closure that count as a microsleep |
| `PERCLOS_WINDOW_S` | 10 | Rolling window for PERCLOS |
| `PERCLOS_ALERT_PCT` | 35.0 | PERCLOS % that triggers a drowsiness alert |
| `MAR_THRESHOLD` | 0.65 | Mouth-opening threshold for a yawn |
| `YAWN_MIN_S` | 1.0 | Minimum yawn duration |
| `YAWN_RATE_ALERT` | 3 | Yawns per minute that trigger an alert |
| `HEAD_PITCH_THRESH` | 20° | Head-down angle for distraction |
| `HEAD_YAW_THRESH` | 35° | Head-turn angle for distraction |
| `HEAD_POSE_MIN_S` | 2.0 | How long a head pose must persist before alerting |
| `NO_FACE_ALERT_S` | 5.0 | Seconds without a face before alerting |
| `ALERT_COOLDOWN_S` | 15 | Minimum gap between standard alarms |
| `FRAME_W` × `FRAME_H` | 640 × 480 | Capture resolution |

---

## 📊 Fatigue Scoring

Six sub-scores (each 0–100) are combined with a weighted sum and smoothed over the last 10 frames:

| Component | Weight |
|---|---|
| PERCLOS | 0.30 |
| EAR deviation from threshold | 0.20 |
| Blink rate | 0.15 |
| Yawn rate | 0.15 |
| Slow-blink ratio | 0.10 |
| Head pose | 0.10 |

A **microsleep bonus** (+30 to +50) is added when the eyes stay closed for ≥ 2.5 s.

| Score | Level | Meaning |
|---|---|---|
| 0 – 30 | 🟢 GREEN | Alert |
| 30 – 50 | 🟡 YELLOW | Early signs of fatigue |
| 50 – 75 | 🟠 ORANGE | Fatigued: voice warning |
| 75 – 100 | 🔴 RED | Critical: urgent voice warning |

### Alert types

- **Microsleep** – critical alarm + voice, bypasses the cooldown
- **PERCLOS ≥ 35%** – drowsiness alert
- **High yawn rate** – ≥ 3 yawns/min
- **Head down / looking away** – persistent for ≥ 2 s
- **Driver not visible** – no face for ≥ 5 s

---

## 🤖 ML Model Integration

Version 3.0 loads a trained scikit-learn-compatible classifier with `joblib` and predicts one of four classes (`0=GREEN`, `1=YELLOW`, `2=ORANGE`, `3=RED`) on every frame.

**Feature vector (26 features):**

| Group | Features |
|---|---|
| Eyes | `ear`, `ear_left`, `ear_right`, `ear_deviation`, `ear_deviation_norm`, `perclos`, `microsleep_s` |
| Blinks | `blink_rate`, `slow_blink_ratio`, `mean_blink_duration_ms` |
| Mouth | `mar`, `yawns_per_minute`, `is_yawning` |
| Head pose | `pitch`, `yaw`, `roll`, `abs_yaw`, `pitch_norm`, `yaw_norm` |
| Sub-scores | `sub_perclos`, `sub_ear`, `sub_blink`, `sub_slow_blink`, `sub_yawn`, `sub_pose`, `fatigue_score` |

**Behaviour:**

- The HUD shows the ML label with its confidence (`predict_proba`) and an **AGREE / CONFLICT** flag against the rule-based level.
- When both engines agree on ORANGE or RED, the level is treated as ML-confirmed.
- Logistic Regression models are automatically fed scaled features (detected via `meta.pkl`); tree-based models receive raw features.
- Every logged alert includes `ml_label` and `ml_confidence`.

### Model performance

> Fill in after training:

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Random Forest | _TBD_ | _TBD_ |
| XGBoost | _TBD_ | _TBD_ |
| Logistic Regression | _TBD_ | _TBD_ |

**Training data:** _describe how the training data was collected / generated here._

---

## 🗂️ Logging and Outputs

Each alert produces:

- **Snapshot:** `drowsy_snapshots/alert_YYYYMMDD_HHMMSS.jpg`
- **CSV row:** `sessions/session_YYYYMMDD_HHMMSS.csv`
- **SQLite row:** `sessions/events.db` → table `events`

**Columns:** `timestamp, ear, mar, fatigue_score, level, alert_type, pitch, yaw, blink_rate, yawn_count, snapshot_path` plus `ml_label, ml_confidence` in v3.0.

Logging runs on background threads so it never blocks the video loop.

---

## ⚠️ Known Limitations

- Accuracy drops with **sunglasses**, heavy glare, or very low light (use `L` for CLAHE).
- Thresholds are calibrated per session; the driver must keep their eyes open and relaxed during the calibration phase.
- Yawn detection can false-trigger during loud talking or laughing.
- A single frontal camera is assumed; extreme camera angles reduce head-pose accuracy.
- This is a **prototype / research project** and is **not a certified automotive safety system**. Do not rely on it as the sole safeguard while driving.

---

## 🚀 Future Work

- Temporal models (LSTM / 1D-CNN) over feature sequences
- Infrared camera support for night driving
- Mobile / embedded deployment (Raspberry Pi, Jetson Nano)
- Fleet dashboard using the SQLite event store
- Evaluation on public drowsiness datasets (e.g. NTHU-DDD, UTA-RLDD)

---

## 🛠️ Tech Stack

Python 3.10+ · MediaPipe 0.10.14 · OpenCV · NumPy · SciPy · scikit-learn · XGBoost · joblib · Pygame · pyttsx3 · SQLite

---

## 📄 License

Add your preferred license here (e.g. MIT).

---

## 👤 Author

**[Your Name]** · [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)
