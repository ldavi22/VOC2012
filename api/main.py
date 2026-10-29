import io
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw,ImageFont


MODEL_DIR = "best_fp32_480.onnx"
sess = ort.InferenceSession(MODEL_DIR,providers=["CPUExecutionProvider"])
CONF = 0.5
INPUT_NAME = sess.get_inputs()[0].name
SIZE = 480
NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
         "train", "tvmonitor"]

def letterbox(img):
    w_i,h_i = img.size
    r = SIZE/max(w_i,h_i)
    w_f = int(np.round(r*w_i)) ; h_f = int(np.round(r*h_i))
    img = img.resize((w_f,h_f),Image.BILINEAR)
    px = ( SIZE - w_f ) // 2 ; py = ( SIZE - h_f ) // 2;
    canvas = Image.new("RGB",(SIZE,SIZE), (114,114,114))
    canvas.paste(img, (px,py))
    return canvas,px,py,r


def to_input(canvas):
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

def infer(x):
    return sess.run(None,{INPUT_NAME: x})[0]

def postprocess(out,r,px,py,W,H,conf):
    det = out[0]
    det = det[det[:,4] > conf]
    boxes = det[:,:4].copy()
    boxes[:,[0,2]] = np.clip((boxes[:,[0,2]] - px ) / r , 0 , W)
    boxes[:,[1,3]] = np.clip((boxes[:,[1,3]] - py ) / r , 0 , H)
    return [
        {"label": NAMES[int(class_num)], "confidence" : round(float(score),3), "box" : [round(float(v), 1) for v in bb]}
        for bb,score,class_num in zip(boxes,det[:,4],det[:,5])
    ]

def draw(img,detections):
    img = img.copy()
    W,H = img.size
    image = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10, int(max(W,H)/100)))  # scalable font
    for d in detections:
        x1,y1,x2,y2 = d["box"]
        text = f"{d['label']} {d['confidence']:.2f}"
        image.rectangle((x1,y1,x2,y2),outline=(255,0,0),width = max(1, int(1/300 * max(W,H))))
        xy = image.textbbox((x1,y1),text,font=font)
        image.rectangle(xy,fill=(255,0,0))
        image.text((x1,y1),text,fill=(255,255,255),font=font)
    return img

def detect(data,conf):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid file")
    W,H = img.size

    t0 = time.perf_counter()
    canvas,px,py,r = letterbox(img)
    yolo_out = infer(to_input(canvas))
    detections = postprocess(yolo_out,r,px,py,W,H,conf)
    ms = (time.perf_counter() - t0) * 1000
    return img,detections,ms

app = FastAPI(title="Object detection with YOLO")

@app.get("/health")
def health():
    return{"status":"ok", "model":MODEL_DIR, "input_size":SIZE, "classes": len(NAMES)}

@app.post("/predict")
def predict(file:UploadFile, conf: float = CONF):
    img,detections,ms = detect(file.file.read(),conf)
    buf = io.BytesIO()
    draw(img,detections).save(buf,format="PNG")
    headers = {"X-Inference-Time-ms": f"{ms:.1f}", "Number-of-detections": str(len(detections))}
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

@app.post("/predict/json")
def predict_json(file:UploadFile,conf: float = CONF):
    img,detections,ms = detect(file.file.read(),conf)
    return {"image_size" : list(img.size) , "time_ms": round(ms) , "detections" : detections}