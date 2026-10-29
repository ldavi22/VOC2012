"""Compare the YOLO variants on VOC val: accuracy, speed, size. Saves results.csv and comparison.png."""

import csv
import os

import matplotlib.pyplot as plt
from ultralytics import YOLO

DATA = "/mnt/mydrive/Uni/dataset/yolo_voc/data.yaml"
MODELS = "/home/ldavi/VOC2012/"  # folder with the four model files

VARIANTS = [  # (name, file, imgsz it was exported with)
    ("PyTorch FP32 640", "best.pt", 640),
    ("ONNX FP32 640", "best_fp32_640.onnx", 640),
    ("ONNX FP32 480", "best_fp32_480.onnx", 480),
    ("ONNX INT8 640", "best_int8_640.onnx", 640),
]

rows = []
for name, file, imgsz in VARIANTS:
    path = os.path.join(MODELS, file)
    m = YOLO(path, task="detect").val(
        data=DATA,
        split="val",
        imgsz=imgsz,
        batch=1,
        rect=False,
        device="cpu",
        plots=False,
        verbose=False,
        nms=False,
    )
    rows.append(
        {
            "model": name,
            "size_mb": os.path.getsize(path) / 1e6,
            "ms_per_image": sum(
                m.speed.values()
            ),  # preprocess + inference + postprocess
            "map50_95": m.box.map,
            "map50": m.box.map50,
        }
    )

base = rows[0]
print(
    f"\n{'model':<18}{'size MB':>9}{'ms/img':>9}{'speedup':>9}{'mAP50-95':>10}{'Δ mAP':>8}{'mAP50':>8}"
)
for r in rows:
    print(
        f"{r['model']:<18}{r['size_mb']:>9.1f}{r['ms_per_image']:>9.1f}"
        f"{base['ms_per_image'] / r['ms_per_image']:>8.2f}x{r['map50_95']:>10.3f}"
        f"{r['map50_95'] - base['map50_95']:>+8.3f}{r['map50']:>8.3f}"
    )

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

names = [r["model"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar(names, [r["ms_per_image"] for r in rows])
ax1.set_title("CPU time per image (ms, lower is better)")
ax2.bar(names, [r["map50_95"] for r in rows])
ax2.set_title("mAP50-95 on val (higher is better)")
for ax in (ax1, ax2):
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
print("\nsaved results.csv and comparison.png")
