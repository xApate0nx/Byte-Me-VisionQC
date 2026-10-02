from pathlib import Path

from app.ml.inference_adapter import inspect_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "defects"
    / "WhatsApp Image 2026-10-02 at 13.51.07.jpeg"
)


result = inspect_image(TEST_IMAGE)

print()
print("========================================")
print("VisionQC PatchCore Backend Test")
print("========================================")
print(f"Image:          {TEST_IMAGE.name}")
print(f"Anomaly score:  {result['anomaly_score']:.6f}")
print(f"Threshold:      {result['threshold']:.6f}")
print(f"Decision:       {result['decision']}")
print(f"ROI:            {result['roi_path']}")

if result["anomaly_map"] is not None:
    print(
        "Anomaly map:    "
        f"{result['anomaly_map'].shape}"
    )
else:
    print("Anomaly map:    None")

print("========================================")