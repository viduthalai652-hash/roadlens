"""RoadLens: stateless ONNX frame inference for Vercel and local use."""
import asyncio
import ast
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'best.onnx'
Image.MAX_IMAGE_PIXELS = 24_000_000
app = FastAPI(title='RoadLens')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
lock = threading.Lock()
session = None
names = {}


def get_model():
    global session, names
    if session is None:
        if not MODEL.is_file():
            raise RuntimeError('Model missing: models/best.onnx must be included in the deployment.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        loaded = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
        metadata = loaded.get_modelmeta().custom_metadata_map
        parsed = ast.literal_eval(metadata.get('names', '{}'))
        parsed = dict(enumerate(parsed)) if isinstance(parsed, list) else parsed
        class_names = {int(k): str(v) for k, v in parsed.items()}
        if 'pothole' not in [v.lower() for v in class_names.values()]:
            raise RuntimeError('The exported model has no pothole class metadata.')
        names = class_names
        session = loaded
    return session


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.


def merge_boxes(candidates, threshold=.5, limit=300):
    kept = []
    for b in sorted(candidates, key=lambda b: b['confidence'], reverse=True):
        if all(overlap(b['xyxy'], k['xyxy']) <= threshold for k in kept):
            kept.append(b)
            if len(kept) >= limit:
                break
    return kept


def preprocess(image, size):
    w, h = image.size
    scale = min(size/w, size/h)
    nw, nh = round(w*scale), round(h*scale)
    left, top = round((size-nw)/2-.1), round((size-nh)/2-.1)
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    padded = Image.new('RGB', (size, size), (114, 114, 114))
    padded.paste(resized, (left, top))
    tensor = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1)[None] / 255.
    return np.ascontiguousarray(tensor), scale, left, top


def decode(output, confidence, scale, left, top, width, height):
    # Export uses raw one-to-many output (1, 4+classes, anchors), retaining
    # the same detection head as the original default Ultralytics prediction.
    raw = np.asarray(output)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4+len(names):
        raise RuntimeError(f'Unexpected ONNX output {raw.shape}. Re-export with nms=None.')
    rows = raw[0].T
    class_ids = rows[:, 4:].argmax(axis=1)
    scores = rows[:, 4:].max(axis=1)
    valid = np.isfinite(rows).all(axis=1) & (scores >= confidence)
    candidates = []
    for r, cls, score in zip(rows[valid], class_ids[valid], scores[valid]):
        if names[int(cls)].lower() != 'pothole':
            continue
        cx, cy, bw, bh = map(float, r[:4])
        box = [(cx-bw/2-left)/scale, (cy-bh/2-top)/scale,
               (cx+bw/2-left)/scale, (cy+bh/2-top)/scale]
        box = [max(0., min(float(width), box[0])), max(0., min(float(height), box[1])),
               max(0., min(float(width), box[2])), max(0., min(float(height), box[3]))]
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append({'xyxy': box, 'confidence': float(score)})
    return merge_boxes(candidates, threshold=.7)


def views(image, size, mode):
    yield image, 0, 0, max(size, 960) if mode == 'thorough' else size
    if mode == 'thorough':
        w, h = image.size
        cw, ch = max(1, round(w*.65)), max(1, round(h*.65))
        for y in sorted({0, h-ch}):
            for x in sorted({0, w-cw}):
                yield image.crop((x, y, x+cw, y+ch)), x, y, 640


def detect(image, confidence, imgsz, mode):
    if not .05 <= confidence <= .95 or imgsz not in (416,640,960,1280) or mode not in ('standard','thorough'):
        raise ValueError('Invalid detection settings.')
    started = time.monotonic()
    w, h = image.size
    candidates = []
    passes = 0
    with lock:
        model = get_model()
        for crop, ox, oy, size in views(image, imgsz, mode):
            tensor, scale, left, top = preprocess(crop, size)
            output = model.run(None, {model.get_inputs()[0].name: tensor})[0]
            boxes = decode(output, confidence, scale, left, top, *crop.size)
            passes += 1
            cw, ch = crop.size
            for b in boxes:
                x1, y1, x2, y2 = b['xyxy']
                if passes > 1 and ((ox > 0 and x1 <= 2) or (oy > 0 and y1 <= 2)
                    or (ox+cw < w and x2 >= cw-2) or (oy+ch < h and y2 >= ch-2)):
                    continue
                candidates.append({'xyxy':[x1+ox,y1+oy,x2+ox,y2+oy], 'confidence':b['confidence']})
    return {'boxes':merge_boxes(candidates), 'width':w, 'height':h, 'mode':mode,
            'passes':passes, 'inference_ms':round((time.monotonic()-started)*1000), 'distance':None}


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html', headers={'Cache-Control':'no-cache'})


@app.get('/api/status')
def status():
    return {'model_file_present':MODEL.is_file(), 'model_loaded':session is not None, 'runtime':'ONNX CPU'}


@app.post('/api/frame')
async def frame(file: UploadFile = File(...), confidence: float = Form(.15),
                imgsz: int = Form(640), mode: str = Form('standard')):
    raw = await file.read(3*1024*1024+1)
    if len(raw) > 3*1024*1024:
        raise HTTPException(413, 'Frame exceeds 3 MB. Resize or compress the image.')
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width*opened.height > 24_000_000:
                raise ValueError('Image exceeds 24 megapixels.')
            image = ImageOps.exif_transpose(opened).convert('RGB')
        return await asyncio.to_thread(detect, image, confidence, imgsz, mode)
    except (UnidentifiedImageError, Image.DecompressionBombError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
