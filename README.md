# VisionQC — Intelligent Visual Quality Control System

> **AI-powered visual inspection for automated product quality control**

VisionQC is an AI-assisted visual quality-control system designed to detect deviations in manufactured products using computer vision and anomaly detection.

The system learns what a **normal/good product** looks like from a small set of reference images and evaluates new product images for visual deviations. It combines product detection, canonical ROI extraction, PatchCore anomaly detection, spatial anomaly analysis, heatmap visualization, configurable inspection thresholds, and inspection history.

The current prototype is developed around a **teal/cyan circular water-cap product** and is designed so that the same workflow can be adapted to other industrial products.

---

## 1. Project Overview

Traditional visual inspection can be repetitive, time-consuming, and dependent on human consistency.

VisionQC addresses this by providing an AI-assisted inspection pipeline:

```text
Camera / Image
      ↓
Image Quality Check
      ↓
Product Detection
      ↓
Canonical ROI Extraction
      ↓
PatchCore Anomaly Detection
      ↓
Spatial Anomaly Analysis
      ↓
Decision Engine
      ↓
PASS / REVIEW / FAIL
      ↓
Heatmap + Confidence + Inspection Log
      ↓
Supervisor Dashboard
```

The system learns the visual characteristics of normal products instead of requiring a large manually labelled defect dataset.

### Current Prototype

The current prototype uses:

* 21 normal product images
* 23 defect images for validation/evaluation
* A teal/cyan circular water-cap product
* Canonical 224 × 224 ROI extraction
* PatchCore with a ResNet18 backbone
* Spatial anomaly features
* Live inspection interface
* Configurable pass/fail threshold
* Inspection history
* Daily inspection statistics
* Anomaly heatmap visualization

---

# 2. Key Features

## AI-Based Anomaly Detection

VisionQC uses **PatchCore** to learn the visual characteristics of normal products and identify deviations in new samples.

The current implementation uses:

* ResNet18 backbone
* `layer2` and `layer3` feature layers
* 9 nearest neighbours
* Canonical 224 × 224 product ROI

The system does not require every possible defect to be manually labelled during normal-model training.

---

## Product Detection

Before anomaly detection, the system identifies the actual product in the image.

The detector was designed around the geometry and appearance of the real water-cap dataset rather than relying only on generic circular Hough detection.

It considers:

* HSV colour characteristics
* Lab colour characteristics
* Saturation/value
* Morphological structure
* Contours
* Circularity
* Aspect ratio
* Product centrality
* Occupancy
* Teal/cyan colour ratio
* Alignment

The current detector successfully detected all available evaluation images in the prototype dataset.

---

## Canonical ROI Extraction

Once the product is detected, VisionQC extracts a standardized region around it.

The ROI pipeline:

```text
Original Image
      ↓
Product Bounding Box
      ↓
Square Crop Around Product
      ↓
30% Contextual Padding
      ↓
Boundary Clamping
      ↓
224 × 224 Canonical ROI
```

This reduces the influence of:

* Camera framing
* Background
* Product position
* Different image dimensions

and gives PatchCore a consistent representation of the product.

---

## Anomaly Heatmap

VisionQC generates a spatial anomaly map showing **where the model detects visual deviation**.

The heatmap is generated from PatchCore's anomaly map and overlaid on the canonical product ROI.

This helps a supervisor understand that the system is not simply returning a numerical score, but is also identifying the regions contributing to the anomaly.

---

## PASS / REVIEW / FAIL Decision

The inspection system supports three primary outcomes:

| Result     | Meaning                                                           |
| ---------- | ----------------------------------------------------------------- |
| **PASS**   | Product is within the configured operating range                  |
| **REVIEW** | Score is close to the configured threshold and should be reviewed |
| **FAIL**   | Product shows significant deviation from learned normal           |

There is also:

| Result                 | Meaning                                       |
| ---------------------- | --------------------------------------------- |
| **INSPECTION_INVALID** | Image/product could not be reliably inspected |

The supervisor can change the operating threshold through the Settings interface.

---

## Confidence Indicator

VisionQC displays a confidence-style inspection value based on the distance of the anomaly score from the operating threshold.

**Important:** this value is an inspection-confidence heuristic and is **not currently a calibrated probability of defect**.

---

## Live Camera Inspection

The frontend supports camera-based inspection through the browser's camera API.

The intended workflow is:

```text
Camera
  ↓
Capture Product
  ↓
Upload to Backend
  ↓
AI Inspection
  ↓
Result + Score + Heatmap
```

The frontend is designed to support webcam and mobile-browser camera workflows.

---

## Inspection Logging

Every inspection is stored in the application database.

Recorded information includes:

* Inspection ID
* Filename
* Timestamp
* Anomaly score
* Threshold
* Decision
* Confidence
* Reason
* Image quality status
* Product detection status
* Alignment status
* Original image
* ROI
* Heatmap
* Supervisor feedback

---

## Supervisor Analytics

The application provides inspection statistics including:

* Total inspections
* Passed inspections
* Failed inspections
* Reviews
* Invalid inspections
* Rejection rate

A dedicated analytics page provides a supervisor-oriented view of inspection activity.

---

# 3. Technology Stack

## Machine Learning / Computer Vision

* Python
* PyTorch
* Anomalib
* PatchCore
* OpenCV
* NumPy
* Pillow
* Scikit-learn

## Backend

* Python
* FastAPI
* SQLAlchemy
* SQLite
* Uvicorn

## Frontend

* React
* TypeScript
* Vite
* CSS
* Browser MediaDevices API

## Development

* Git
* GitHub
* Visual Studio Code
* Python virtual environment

---

# 4. Architecture / Workflow

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Camera / Image    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │  Live Inspection    │
                    └──────────┬──────────┘
                               │
                         HTTP / API
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Quality Check │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Product Detection   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Canonical ROI       │
                    │ 224 × 224           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ PatchCore + ResNet18│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌──────────────────┐
          │ Anomaly Score   │   │ Anomaly Map      │
          └────────┬────────┘   └────────┬─────────┘
                   │                     │
                   ▼                     ▼
          ┌─────────────────────────────────────┐
          │       Decision / Spatial Analysis   │
          └──────────────────┬──────────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ PASS / REVIEW / FAIL   │
                 └────────────┬───────────┘
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
           Heatmap        Inspection      Analytics
                          Database        Dashboard
```

---

## Inspection Workflow

### Step 1 — Image Acquisition

A product image is captured using a camera or supplied as an image file.

### Step 2 — Image Quality Validation

The backend checks whether the image can be processed reliably.

### Step 3 — Product Detection

The product detector identifies the water-cap and determines its bounding box and alignment.

### Step 4 — Canonical ROI

The detected product is converted into a standardized 224 × 224 ROI.

### Step 5 — PatchCore

The ROI is passed through the trained PatchCore model.

PatchCore compares the extracted feature representation against the learned normal feature memory bank.

### Step 6 — Spatial Analysis

The anomaly map is analyzed to obtain spatial characteristics such as:

* Central anomaly fraction
* Border anomaly fraction
* Anomaly centroid
* Centroid distance
* Largest anomalous region
* Anomaly mass

### Step 7 — Decision

The anomaly score is compared with the supervisor-configured operating threshold.

The system returns:

```text
PASS
REVIEW
FAIL
```

or `INSPECTION_INVALID` if the inspection itself is unreliable.

### Step 8 — Visualization

The frontend displays:

* Result
* Anomaly score
* Threshold
* Confidence indicator
* Product image
* Anomaly heatmap
* Inspection explanation

### Step 9 — Logging

The inspection is stored for later review and analytics.

---

# 5. Project Structure

```text
visionqc/
│
├── backend/
│   └── app/
│       ├── api/
│       │   ├── inspect.py
│       │   ├── inspections.py
│       │   ├── settings.py
│       │   └── stats.py
│       │
│       ├── database/
│       ├── ml/
│       │   └── inference_adapter.py
│       │
│       ├── services/
│       │   └── inspection_service.py
│       │
│       └── main.py
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       ├── hooks/
│       ├── pages/
│       ├── services/
│       ├── styles/
│       ├── App.tsx
│       └── main.tsx
│
├── vision/
│   ├── product_detector.py
│   ├── cap_roi.py
│   ├── spatial_features.py
│   ├── decision_engine.py
│   └── run_patchcore_inspection.py
│
├── models/
│   └── patchcore_water_cap_v4/
│
├── data/
│   ├── products/
│   ├── evaluation/
│   ├── database/
│   └── inspections/
│
├── requirements.txt
└── README.md
```

---

# 6. Dataset / API Information

## Dataset

The prototype dataset contains images of a single product type: a teal/cyan circular water-cap.

Current dataset:

| Category | Images |
| -------- | -----: |
| Normal   |     21 |
| Defects  |     23 |
| Total    |     44 |

Normal images are used to establish the visual representation of acceptable products.

Defect images are used for evaluation and validation of the prototype.

The dataset is intentionally small because the project is designed around **few-shot/one-class anomaly detection**, where the model primarily learns from normal examples.

### Dataset Limitations

The current dataset is not large enough to establish production-grade statistical performance across:

* Different manufacturing batches
* Different lighting conditions
* Different cameras
* Different product variants
* Large numbers of defect types
* Severe physical deformation
* Environmental changes

Further data collection is required before deployment in an industrial production environment.

---

# 7. Model Information

The current production prototype uses **PatchCore V4**.

### Model Configuration

```text
Backbone:       ResNet18
Feature layers: layer2, layer3
Neighbours:     9
Input:          224 × 224 canonical ROI
Inference:      CPU-compatible
```

The model was trained using canonical ROIs generated from the normal-product images.

### Validation

The final validation experiment used:

* 15 normal images for training
* 6 unseen normal images
* 23 unseen defect images

The experiment demonstrated that the raw PatchCore anomaly score alone does **not provide perfect separation** between the current normal and defect sets.

Therefore, the prototype intentionally treats the PatchCore score as an anomaly signal rather than claiming it is a perfect binary classifier.

This is also why the system provides a **REVIEW** state instead of forcing every borderline inspection into PASS or FAIL.

---

# 8. Backend API

The FastAPI backend exposes endpoints for inspection, settings, statistics, and inspection history.

## Health

```http
GET /api/health
```

Returns:

```json
{
  "status": "ok"
}
```

## Inspection

```http
POST /api/inspect
```

Accepts:

* JPEG
* PNG
* WebP

The image is processed through the complete VisionQC pipeline.

## Threshold

```http
GET /api/settings/threshold
```

Returns the current inspection threshold.

```http
PUT /api/settings/threshold
```

Updates the supervisor-configured threshold.

## Statistics

```http
GET /api/stats/today
```

Returns today's:

* Total inspections
* Passed
* Failed
* Reviews
* Invalid inspections
* Rejection rate

## Inspection History

```http
GET /api/inspections
```

Returns recorded inspection results.

---

# 9. Setup & Installation

## Requirements

Recommended environment:

* Python 3.10+
* Node.js 18+
* npm
* Git

---

## Clone Repository

```bash
git clone https://github.com/xApate0nx/Byte-Me-VisionQC.git
cd Byte-Me-VisionQC
```

---

## Python Environment

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

---

## Start Backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/api/health
```

---

## Start Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on the Vite development server, normally:

```text
http://localhost:5174
```

The frontend is configured to proxy:

```text
/api
/storage
```

to the FastAPI backend.

---

# 10. Running an Inspection Directly

The PatchCore inspection pipeline can also be executed directly from the command line.

Example:

```powershell
python vision\run_patchcore_inspection.py "path\to\image.jpg"
```

A custom threshold can be supplied:

```powershell
python vision\run_patchcore_inspection.py "path\to\image.jpg" 14.0
```

The pipeline performs:

```text
Image
 ↓
Product Detection
 ↓
Canonical ROI
 ↓
PatchCore
 ↓
Anomaly Map
 ↓
Spatial Features
 ↓
Decision
```

---

# 11. Screenshots / Demo

The repository contains frontend sample images under:

```text
frontend/public/samples/
```

including:

```text
normal_cap.jpg
defective_cap.jpg
```

The application interface includes:

### Live Inspection

Provides the main inspection workflow with camera/image capture, product framing, inspection result, anomaly score, and heatmap.

### Inspection Detail

Displays the inspected image, ROI, anomaly visualization, score, threshold, and decision information.

### Analytics

Displays inspection statistics and rejection information.

### Settings

Allows the supervisor to configure the inspection threshold.

### Demo Flow

A typical demonstration is:

```text
1. Open VisionQC
2. Start Live Inspection
3. Present/capture a water-cap
4. Submit the inspection
5. Wait for AI analysis
6. View PASS / REVIEW / FAIL
7. Inspect the anomaly heatmap
8. Open inspection history
9. Adjust threshold from Settings
10. View updated analytics
```

---

# 12. Limitations

The current project is a functional prototype and has several limitations.

### Dataset Size

The current dataset contains only 44 images.

More production data is required for robust generalization.

### Single Product Type

The current model is designed for one water-cap product.

A new product would require its own normal reference dataset and model configuration.

### Lighting and Camera Variability

Large changes in:

* illumination
* camera angle
* distance
* background
* reflections

can affect visual anomaly detection.

### PatchCore Score Overlap

The current validation demonstrates overlap between normal and defect anomaly scores.

Therefore, the raw anomaly score should not be interpreted as a guaranteed defect probability.

### Confidence Calibration

The displayed confidence value is currently a threshold-distance heuristic rather than a statistically calibrated probability.

### Production Hardware

The current implementation is CPU-compatible and suitable for prototyping.

Industrial deployment would benefit from dedicated GPU inference hardware and optimized model serving.

### Mobile Deployment

Browser-based mobile camera integration is part of the intended workflow, but reliable deployment across different mobile browsers, network configurations, and HTTPS environments requires additional testing.

---

# 13. Future Scope

VisionQC can be extended into a larger industrial inspection platform.

## Multi-Product Support

Support multiple products with:

```text
Product
   ↓
Product-specific detector
   ↓
Product-specific ROI
   ↓
Product-specific anomaly model
```

---

## Larger Industrial Dataset

Future versions can incorporate:

* Thousands of normal samples
* Multiple production batches
* Multiple cameras
* Multiple lighting conditions
* Multiple defect categories

---

## Improved Model Calibration

Future work can investigate:

* Threshold calibration
* ROC/PR analysis
* Reliability calibration
* Better confidence estimation
* Larger validation datasets
* Statistical process-control integration

---

## Advanced Anomaly Fusion

Spatial features can be combined with PatchCore scores using a properly validated decision layer once enough data is available.

Potential future features include:

* Learned anomaly fusion
* Temporal inspection trends
* Batch-level anomaly monitoring
* Defect clustering
* Automatic defect categorization

---

## Edge Deployment

The system can eventually be deployed on:

* Industrial PCs
* NVIDIA edge devices
* Factory inspection stations
* Dedicated camera systems

to enable low-latency inspection without relying on cloud inference.

---

## Manufacturing Integration

Future versions could integrate with:

* PLC systems
* Conveyor systems
* Industrial cameras
* Automated rejection mechanisms
* MES/ERP systems
* Production-line dashboards

This would allow VisionQC to progress from an inspection assistant into a complete automated quality-control system.

---

# 14. Team Members

| Member               | Role        |
| -------------------- | ----------- |
| **Ruthvik Narvekar** | Leader      |
| **Vaishnavi Mankar** | Team Member |
| **Hitesh Yelve**     | Team Member |
| **Niharika Sonkar**  | Team Member |

---

# 15. Project Status

**Current status: Functional prototype**

Implemented:

* [x] Normal-product learning pipeline
* [x] Product detection
* [x] Canonical ROI extraction
* [x] PatchCore anomaly detection
* [x] Spatial anomaly analysis
* [x] Heatmap generation
* [x] PASS / REVIEW / FAIL decision flow
* [x] Configurable inspection threshold
* [x] Inspection database
* [x] Inspection history
* [x] Daily analytics
* [x] React frontend
* [x] Live camera workflow
* [x] Backend REST API
* [x] Supervisor settings
* [x] Sample demonstration images

---

# 16. Repository

**GitHub:**
https://github.com/xApate0nx/Byte-Me-VisionQC

VisionQC is developed as a modular system so that the computer-vision pipeline, AI model, backend services, and frontend interface can evolve independently toward a production-ready industrial inspection solution.
