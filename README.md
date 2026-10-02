# VisionQC

VisionQC is a computer-vision prototype for inspecting water bottle caps (`water_cap_v1`) and detecting visual defects. The work completed so far is focused on local Python experiments for image features, anomaly detection, PatchCore training and inference, thresholding, and inspection visualizations.

## Project Layout

The following tree shows the current workspace. Image files and generated model/evaluation artifacts are grouped rather than listed individually; `.venv/` and Python cache files are omitted.

```text
visionqc/
├── .gitignore
├── README.md
├── docker-compose.yml
├── backend/
│   ├── .env
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── inspect.py
│   │   │   ├── inspections.py
│   │   │   ├── settings.py
│   │   │   └── stats.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── constants.py
│   │   ├── database/
│   │   │   ├── database.py
│   │   │   ├── models.py
│   │   │   └── schemas.py
│   │   ├── ml/
│   │   │   └── inference_adapter.py
│   │   └── services/
│   │       ├── inspection_service.py
│   │       ├── storage_service.py
│   │       └── threshold_service.py
│   └── tests/
│       ├── test_inspect.py
│       ├── test_settings.py
│       └── test_stats.py
├── data/
│   ├── products/water_cap_v1/
│   │   ├── normal/                # 21 product images
│   │   ├── validation/            # currently empty
│   │   └── defects/               # 23 product images
│   ├── normal/{train,validation}/
│   ├── defects/test/
│   ├── samples/
│   ├── evaluation/
│   │   ├── roi_candidates/{small,medium,large}/{normal,defects}/
│   │   ├── roi_test/{normal,defects}/
│   │   ├── stable_region/
│   │   ├── patchcore_heatmaps/{normal,defects}/
│   │   ├── normalized_contact_sheet/
│   │   └── Patchcore/water_cap_evaluation/latest/images/{normal,defects}/
│   ├── inspections/
│   │   ├── _temporary_roi/
│   │   ├── heatmaps/
│   │   └── processed/
│   ├── heatmaps/
│   └── inspection/
├── database/                      # planned database package
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── public/
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/
│       │   ├── Camera/
│       │   ├── Heatmap/
│       │   ├── InspectionResult/
│       │   ├── StatsCard/
│       │   └── ThresholdControl/
│       ├── hooks/useCamera.ts
│       ├── pages/{Dashboard,Inspection}.tsx
│       ├── services/api.ts
│       └── types/inspection.ts
├── ml/
│   ├── README.md
│   ├── training/{train.py,config.yaml}
│   ├── inference/{inspect.py,preprocessing.py}
│   └── evaluation/{evaluate.py,results/}
├── models/
│   ├── exported/
│   ├── patchcore/{checkpoint,config}/
│   ├── patchcore_water_cap/
│   │   └── Patchcore/water_cap_v1/{latest,v0/weights/lightning}/
│   ├── patchcore_water_cap_v2/
│   │   └── Patchcore/
│   │       ├── water_cap_v1_roi80/{latest,v0/weights/lightning}/
│   │       ├── water_cap_v1_roi80_evaluation/latest/images/{defects,normal}/
│   │       └── water_cap_v1_roi80_raw/latest/images/{defects,normal}/
│   └── visionqc_resnet18/          # ResNet reference-model output
├── results/
│   └── Patchcore/
│       ├── latest/images/{defects,processed}/
│       └── water_cap_evaluation/latest/images/{defects,normal}/
├── scripts/{setup.py,run_dev.py}
├── storage/
│   ├── inspections/{images,heatmaps}/
│   └── visionqc.db
└── vision/                          # active Python prototype scripts
	├── analyze_image_composition.py
	├── analyze_patchcore_maps.py
	├── anomaly_detector.py
	├── calibrate_patchcore_threshold.py
	├── create_normalized_contact_sheet.py
	├── create_roi_candidates.py
	├── decision_engine.py
	├── evaluate_patchcore.py
	├── evaluate_patchcore_v2.py
	├── evaluate_patchcore_v2_raw.py
	├── feature_extractor.py
	├── find_stable_product_region.py
	├── generate_inspection_heatmap.py
	├── inference.py
	├── inspect_dataset.py
	├── inspect_patchcore_heatmaps.py
	├── local_heatmap.py
	├── resnet_anomaly_detector.py
	├── run_patchcore_auto.py
	├── run_patchcore_inspection.py
	├── test_roi_detection.py
	├── train_patchcore.py
	└── train_patchcore_v2.py
```

## Implemented So Far

- **Classical baseline:** `vision/anomaly_detector.py` extracts color histograms and downsampled grayscale pixels, trains an `IsolationForest` on normal images, and reports predictions for the normal and defect folders.
- **ResNet-18 feature experiments:** `vision/feature_extractor.py` extracts pretrained ResNet-18 embeddings. `vision/inference.py` builds a median reference feature map from normal images, scores feature-map deviations, estimates a coarse anomaly region, and can save a heatmap and JSON result. The reference uses the `layer4` feature map.
- **PatchCore experiments:** `vision/train_patchcore.py` and the v2 scripts train/evaluate PatchCore with a ResNet-18 backbone. The v2 training script uses `layer2` and `layer3`, expects 21 normal images in `data/evaluation/roi_candidates/large/normal`, and writes model output under `models/patchcore_water_cap_v2/`.
- **Inspection decisions:** `vision/decision_engine.py` converts an anomaly score into `PASS`, `REVIEW`, or `FAIL`, and can return `INSPECTION_INVALID` when a quality, product-detection, alignment, or score check fails. Confidence is a score-distance heuristic, not a probability.
- **Inspection artifacts:** `vision/run_patchcore_inspection.py` runs PatchCore on one ROI image, saves its anomaly map as a NumPy file, applies the decision engine, and writes an inspection JSON record. `vision/generate_inspection_heatmap.py` renders a saved anomaly map over its source image.
- **Dataset utilities:** scripts in `vision/` create contact sheets, inspect PatchCore maps, examine image composition, and explore candidate/stable product regions.

## Current Data

The product-specific image folders are under `data/products/water_cap_v1/`:

- `normal/`: 21 images currently present.
- `defects/`: 23 images currently present.
- `validation/`: currently empty.

The PatchCore v2 scripts use a separate ROI dataset under `data/evaluation/roi_candidates/large/`, with `normal/` and `defects/` subfolders. Keep the training and evaluation images organized as expected by those scripts.

## Running The Scripts

Run commands from the project root. Install compatible versions of the packages imported by the selected script first; dependencies are not pinned in `backend/requirements.txt` yet. The vision scripts use packages including PyTorch, torchvision, anomalib, NumPy, Pillow, Matplotlib, OpenCV, and scikit-learn.

Train the PatchCore v2 model:

```powershell
python vision/train_patchcore_v2.py
```

Evaluate raw PatchCore scores on the configured ROI dataset:

```powershell
python vision/evaluate_patchcore_v2_raw.py
```

Run a PatchCore inspection on one ROI image:

```powershell
python vision/run_patchcore_inspection.py "data/evaluation/roi_candidates/large/defects/example.jpg"
```

Generate a visual heatmap from the image and anomaly map paths printed by the inspection script:

```powershell
python vision/generate_inspection_heatmap.py "path/to/image.jpg" "data/inspections/heatmaps/example_anomaly_map.npy"
```

The PatchCore training, evaluation, and inspection scripts are configured to use the CPU. The ResNet feature-map experiment selects CUDA when available. Check the script paths and directories if your local data layout differs.

## Threshold Status

`vision/calibrate_patchcore_threshold.py` compares stored raw scores for 10 normal and 23 defect samples. The configured PatchCore decision threshold is `12.3785223961`, the midpoint between the highest listed normal score (`12.3268375397`) and the lowest listed defect score (`12.4302072525`). This separates those recorded samples, but it is not a production threshold: it was selected from a small evaluation set and needs independent validation on more data.

The end-to-end PatchCore runner currently passes image quality, product detection, and alignment as valid. The decision engine supports invalid states, but real camera checks, product detection, and alignment measurement are not yet wired into the inspection flow.

## Project Status

The active implementation is in `vision/`. The `backend/`, `frontend/`, top-level `ml/`, database, and scripts directories currently contain the planned scaffold, not a functioning API, web interface, training package, or database integration. The frontend package manifest is also not yet configured with dependencies or run scripts.
