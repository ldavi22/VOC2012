import shutil
from ultralytics import YOLO

MODEL_PATH = "/kaggle/input/models/ldavi22/dasda/pytorch/default/1/best.pt"

shutil.copy("/kaggle/input/models/ldavi22/dasda/pytorch/default/1/best.pt", "/kaggle/working/best.pt")

variants = {
    "best_fp32_640.onnx": dict(imgsz=640, nms=False),
    "best_fp32_480.onnx": dict(imgsz=480, nms=False),
    "best_int8_640.onnx": dict(imgsz=640, nms=False, quantize=8,
                               data="/kaggle/working/yolo_voc/data.yaml", split="train", fraction=0.1),
}
for name, kwargs in variants.items():
    path = YOLO("/kaggle/working/best.pt").export(format="onnx", **kwargs)
    shutil.move(path, f"/kaggle/working/{name}")
    print(name, "done")