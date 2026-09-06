# ⚾ BaseVision AI

**AI-Powered Baseball Performance & Scouting System**

BaseVision AI is a computer vision and machine learning project designed to analyze baseball video and extract useful performance, motion, and scouting information automatically.

The project combines custom object detection, ball tracking, player role detection, pose estimation, swing analytics, and heuristic event detection into a modular baseball video analysis pipeline.

---

## 🚀 Project Overview

BaseVision AI processes baseball broadcast footage and transforms raw video into structured analytics.

The current pipeline can:

- Detect the baseball using a custom YOLO model
- Detect and identify key baseball roles
- Track the ball across frames
- Estimate batter body pose
- Smooth noisy pose keypoints
- Measure swing-related motion
- Detect swing phases
- Identify candidate contact windows
- Analyze ball trajectory and motion changes
- Export structured CSV results for future dashboards and machine learning models

The long-term goal is to build a complete scouting platform capable of generating advanced baseball analytics from video.

---

## 🧠 Current Computer Vision Pipeline

```text
Baseball Video
      │
      ▼
OpenCV Video Processing
      │
      ▼
YOLO Object Detection
      │
      ├── Baseball
      └── Players
            │
            ▼
      Role Detection
      ├── Pitcher
      ├── Batter
      ├── Catcher
      └── Umpire
            │
            ▼
      Batter Tracking
            │
            ▼
      Pose Estimation
            │
            ▼
      Pose Smoothing
            │
            ▼
      Swing Analytics
            │
            ▼
      Swing Phase Detection
            │
            ▼
      Contact Candidate Window

Ball Detection
      │
      ▼
Kalman Ball Tracking
      │
      ▼
Trajectory Analysis
      │
      ▼
Pitch / Contact Analysis
```

---

## ✅ Current Features

### ⚾ Baseball Detection

A custom YOLO model was trained specifically to detect a baseball in broadcast footage where the ball is:

- Very small
- Motion blurred
- Fast moving
- Partially occluded
- Difficult to detect with generic pretrained models

Current baseline model:

```text
Model: YOLO Nano
Image size: 1280
Task: Baseball detection
Classes: 1
Class: baseball
```

Baseline validation results:

| Metric | Result |
|---|---:|
| Precision | 0.527 |
| Recall | 0.667 |
| mAP@50 | 0.524 |
| mAP@50-95 | 0.278 |

> These results correspond to an early baseline trained on a small dataset. The current dataset contains frames from a limited number of video sequences, so these metrics should not yet be interpreted as generalization performance.

---

## 🎯 Ball Tracking

The system combines baseball detections with motion-based tracking.

Several tracking approaches were tested during development:

- Highest-confidence detection
- Distance-gated tracking
- Constant-velocity prediction
- Kalman filtering

The current experimental tracker uses a Kalman filter to estimate ball position when YOLO temporarily loses the baseball.

Example tracking states:

```text
WAITING
TRACKING
KALMAN
LOST
RESET
```

This allows the pipeline to maintain short-term trajectory continuity even during temporary detection gaps.

---

## 👤 Player Detection and Role Assignment

A generic YOLO person detector is combined with baseball-field spatial heuristics to identify key players.

Currently supported roles:

- Pitcher
- Batter
- Catcher
- Umpire

The system uses:

- Field-region filtering
- Position constraints
- Home plate region
- Pitcher region
- Temporal consistency
- Bounding-box proximity

This helps reduce false detections from spectators and other people visible in broadcast footage.

---

## 🏏 Batter Tracking

After detecting the batter, BaseVision AI maintains a stabilized batter bounding box across frames.

The tracker uses:

- Temporal bounding-box consistency
- Exponential moving average smoothing
- Maximum center displacement
- Short-term missing-frame recovery

Possible tracking states include:

```text
TRACKING
HOLD
REJECTED_JUMP
LOST
WAITING
```

---

## 🧍 Pose Estimation

The batter is processed using a YOLO pose model.

The pose pipeline estimates 17 COCO keypoints:

```text
nose
left_eye
right_eye
left_ear
right_ear
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
```

Pose estimation is performed inside the tracked batter ROI to reduce interference from other players.

---

## 🧹 Pose Smoothing

Raw pose keypoints can become noisy because of:

- Motion blur
- Occlusion
- Fast swing motion
- Partial body visibility
- Broadcast compression

The smoothing stage applies:

- Confidence filtering
- Short-gap interpolation
- Exponential moving average smoothing
- Pose quality estimation

This produces a more stable motion representation for swing analysis.

---

## 🏏 Swing Analytics

BaseVision AI extracts several 2D projected swing metrics.

Current measurements include:

- Left wrist speed
- Right wrist speed
- Wrist speed relative to shoulder translation
- Hip center movement
- Shoulder center movement
- Shoulder orientation
- Hip orientation
- Shoulder-to-hip separation
- Knee height difference
- Ankle height difference
- Leg kick activity

Example results from the current test sequence:

```text
Maximum left wrist speed:
66.01 px/frame

Maximum right wrist speed:
74.29 px/frame

Maximum left wrist relative speed:
67.28 px/frame

Maximum right wrist relative speed:
76.86 px/frame
```

The strongest hand-motion event in the current test sequence occurs around frame 102.

> These values are image-plane measurements in pixels per frame. They are not equivalent to physical bat speed or real-world velocity without camera calibration.

---

## 🔄 Swing Phase Detection

BaseVision AI includes a heuristic swing phase detector.

Current phases:

```text
SETUP
   ↓
LOAD
   ↓
SWING
   ↓
PEAK_SWING
   ↓
FOLLOW_THROUGH
```

The phase detector uses a combination of:

- Wrist relative motion
- Hip movement
- Shoulder movement
- Leg activity
- Temporal progression

The current system also identifies a:

```text
CONTACT_CANDIDATE
```

window near peak swing activity.

This is intentionally treated as a candidate rather than confirmed contact.

---

## ⚾ Pitch and Contact Analysis

The ball trajectory module analyzes tracked baseball motion.

Current experimental measurements include:

- Horizontal displacement
- Vertical displacement
- Average pixel velocity
- Motion direction
- Local trajectory angle
- Local trajectory change
- Speed change
- Track transition timing

A heuristic contact score evaluates whether the trajectory changes strongly enough to suggest ball-bat contact.

The current implementation does not force a contact event when the evidence is weak.

---

## 📁 Project Structure

```text
BaseVisionAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset/
│
├── models/
│
├── src/
│   │
│   ├── detection/
│   │   ├── detect_video.py
│   │   ├── detect_players.py
│   │   ├── detect_players_v2.py
│   │   ├── extract_frames.py
│   │   ├── test_baseball_model.py
│   │   └── train_baseball.py
│   │
│   ├── ball/
│   │   ├── track_ball.py
│   │   ├── track_ball_v2.py
│   │   ├── track_ball_v2_1.py
│   │   ├── track_ball_v3_kalman.py
│   │   └── track_ball_v3_1.py
│   │
│   ├── pitch/
│   │   ├── analyze_pitch.py
│   │   └── detect_contact.py
│   │
│   └── swing/
│       ├── track_batter.py
│       ├── batter_pose.py
│       ├── smooth_pose.py
│       ├── analyze_swing.py
│       └── detect_swing_phases.py
│
├── dashboard/
│
├── notebooks/
│
├── reports/
│
├── results/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

### Computer Vision

- Python
- OpenCV
- Ultralytics YOLO
- YOLO Pose
- Kalman Filtering

### Deep Learning

- PyTorch
- CUDA
- NVIDIA GPU acceleration

### Data Analysis

- NumPy
- Pandas
- Matplotlib

### Planned Analytics & ML

- scikit-learn
- XGBoost
- CatBoost
- Plotly
- Streamlit

### Planned Deployment

- FastAPI
- Docker

---

## 💻 Development Environment

Current development environment:

```text
Operating System:
Windows

Python:
3.10

GPU:
NVIDIA RTX 4060 8 GB

Deep Learning Framework:
PyTorch + CUDA

Computer Vision:
OpenCV + Ultralytics YOLO
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rob437ertoU/BaseVisionAI.git
```

Enter the project directory:

```bash
cd BaseVisionAI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Example Workflow

The current development pipeline can be executed step by step.

### 1. Train the baseball detector

```bash
python src/detection/train_baseball.py
```

### 2. Test the custom baseball model

```bash
python src/detection/test_baseball_model.py
```

### 3. Detect baseball players

```bash
python src/detection/detect_players_v2.py
```

### 4. Track the baseball

```bash
python src/ball/track_ball_v3_1.py
```

### 5. Analyze the ball trajectory

```bash
python src/pitch/analyze_pitch.py
```

### 6. Evaluate possible contact

```bash
python src/pitch/detect_contact.py
```

### 7. Track the batter

```bash
python src/swing/track_batter.py
```

### 8. Estimate batter pose

```bash
python src/swing/batter_pose.py
```

### 9. Smooth pose keypoints

```bash
python src/swing/smooth_pose.py
```

### 10. Analyze the swing

```bash
python src/swing/analyze_swing.py
```

### 11. Detect swing phases

```bash
python src/swing/detect_swing_phases.py
```

---

## 📊 Generated Outputs

The system produces structured outputs such as:

```text
results/
│
├── ball_tracking_v3_1.csv
├── pitch_analysis_summary.csv
├── pitch_analysis_points.csv
├── contact_detection.csv
├── player_detection_v2.csv
├── batter_tracking_v1.csv
├── batter_pose_v1.csv
├── batter_pose_smoothed_v1.csv
├── swing_analysis_v1_3.csv
├── swing_analysis_summary_v1_3.csv
├── swing_phases_v1.csv
└── swing_phases_summary_v1.csv
```

Generated videos and analysis files are excluded from Git tracking to keep the repository lightweight.

---

## ⚠️ Current Limitations

BaseVision AI is currently an experimental research and portfolio project.

Important limitations include:

- The custom baseball dataset is still small
- Early training data contains adjacent frames from limited video sequences
- Current model metrics should be treated as baseline results
- Ball tracking can still fail during severe blur or occlusion
- Player role assignment currently uses scene-specific spatial heuristics
- Pose estimation is 2D and affected by camera perspective
- Pixel velocity is not equivalent to real-world speed
- Physical pitch speed requires camera calibration
- Swing metrics are projected image-plane measurements
- Contact detection is currently heuristic
- The system has not yet been validated across multiple stadiums and camera angles

---

## 🗺️ Roadmap

### Phase 1 — Computer Vision Core

- [x] Video processing with OpenCV
- [x] Pretrained YOLO testing
- [x] Baseball dataset creation
- [x] Custom baseball detector
- [x] Ball tracking
- [x] Player detection
- [x] Batter tracking
- [x] Batter pose estimation
- [x] Pose smoothing

### Phase 2 — Baseball Analytics

- [x] Ball trajectory analysis
- [x] Basic contact heuristic
- [x] Swing motion analysis
- [x] Swing phase detection
- [ ] Swing phase video visualization
- [ ] Multi-video validation
- [ ] Camera calibration
- [ ] Approximate physical ball velocity
- [ ] Improved contact detection

### Phase 3 — Machine Learning

- [ ] Pitch type classification
- [ ] Swing outcome classification
- [ ] Contact probability model
- [ ] Hit probability prediction
- [ ] Exit trajectory analysis
- [ ] Batter performance modeling

### Phase 4 — Scouting Platform

- [ ] Interactive Streamlit dashboard
- [ ] Plotly visualizations
- [ ] Player scouting profiles
- [ ] Automatic game analysis
- [ ] Automatic scouting reports
- [ ] PDF report generation

### Phase 5 — Deployment

- [ ] FastAPI backend
- [ ] Docker deployment
- [ ] Cloud inference
- [ ] Web-based video upload
- [ ] End-to-end automated processing

---

## 🎯 Long-Term Vision

The final objective of BaseVision AI is to transform baseball video into structured scouting information.

Future versions aim to estimate metrics such as:

```text
Pitch
├── trajectory
├── approximate velocity
├── movement
└── pitch classification

Swing
├── timing
├── load
├── acceleration
├── peak swing
├── follow-through
└── contact window

Batted Ball
├── trajectory
├── launch direction
├── approximate exit angle
└── outcome prediction

Scouting
├── player tendencies
├── swing profiles
├── pitch response
└── performance prediction
```

---

## 📈 Future Machine Learning Architecture

```text
Video
  │
  ▼
Computer Vision
  │
  ├── Ball Detection
  ├── Player Detection
  ├── Pose Estimation
  └── Tracking
  │
  ▼
Feature Engineering
  │
  ├── Ball trajectory
  ├── Wrist motion
  ├── Hip motion
  ├── Swing timing
  ├── Body rotation
  └── Contact candidate
  │
  ▼
Machine Learning
  │
  ├── Pitch classification
  ├── Contact prediction
  ├── Hit prediction
  └── Scouting models
  │
  ▼
Interactive Dashboard
```

---

## 📌 Project Status

**Current status: Active Development**

BaseVision AI currently has a functional experimental pipeline covering:

```text
Video
→ Baseball Detection
→ Ball Tracking
→ Player Detection
→ Batter Tracking
→ Pose Estimation
→ Pose Smoothing
→ Swing Analytics
→ Swing Phase Detection
→ Pitch / Contact Analysis
```

The next development focus is improving visualization, multi-video validation, and integrating the current modules into a unified analytics pipeline.

---

## 👨‍💻 Author

**Roberto Eduardo Cardona Luis**

Engineering in Robotics and Digital Systems

Interests:

- Computer Vision
- Artificial Intelligence
- Robotics
- Machine Learning
- Autonomous Systems
- Sports Analytics

---

## 📄 License

This project is currently intended for educational, research, and portfolio purposes.

Dataset licensing and third-party model licensing should be reviewed before commercial use.

---

## ⭐ BaseVision AI

> Turning baseball video into structured computer vision and scouting analytics.
