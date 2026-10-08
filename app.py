"""Local-only web interface for existing pothole weights."""
import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
app = FastAPI(title='Road Hazard Local Website')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
lock = threading.Lock()
model = None
jobs = {}
executor = ThreadPoolExecutor(max_workers=1)
MAX_BYTES = 300 * 1024 * 1024


def get_model():
    global model
    if model is None:
        path = Path(os.environ.get('POTHOLE_WEIGHTS', str(ROOT / 'models' / 'best.pt')))
        if not path.is_file():
            raise RuntimeError('Model missing. Copy your trained best.pt into this project’s models folder.')
        model = YOLO(str(path))
        if 'pothole' not in [str(n).lower() for n in model.names.values()]:
            model = None
            raise RuntimeError('The supplied model has no pothole class.')
    return model


def validate_settings(confidence, imgsz, mode):
    if not 0.05 <= confidence <= .95 or imgsz not in (416, 640, 960, 1280):
        raise ValueError('Invalid confidence or inference size.')
    if mode not in ('standard', 'thorough'):
        raise ValueError('Invalid analysis mode.')


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def merge_boxes(candidates):
    # Suppress duplicate observations across crops; retain highest-confidence box.
    kept = []
    for candidate in sorted(candidates, key=lambda b: b['confidence'], reverse=True):
        if all(overlap(candidate['xyxy'], b['xyxy']) <= .5 for b in kept):
            kept.append(candidate)
    return kept


def views(frame, imgsz, mode):
    yield frame, 0, 0, max(imgsz, 960) if mode == 'thorough' else imgsz
    if mode == 'thorough':
        h, w = frame.shape[:2]
        # Four overlapping crops expose detail lost when the full image is resized.
        ch, cw = max(1, round(h*.65)), max(1, round(w*.65))
        for y in sorted({0, h-ch}):
            for x in sorted({0, w-cw}):
                yield frame[y:y+ch, x:x+cw], x, y, 640


def detect(frame, confidence, imgsz, mode='standard'):
    validate_settings(confidence, imgsz, mode)
    started = time.monotonic()
    h, w = frame.shape[:2]
    candidates = []
    passes = 0
    with lock:
        detector = get_model()
        for image, ox, oy, size in views(frame, imgsz, mode):
            result = detector.predict(image, conf=confidence, imgsz=size,
                                      iou=.7, max_det=300, verbose=False)[0]
            passes += 1
            vh, vw = image.shape[:2]
            for b in result.boxes:
                if str(detector.names[int(b.cls.item())]).lower() != 'pothole':
                    continue
                local = list(map(float, b.xyxy[0].tolist()))
                # Crop-edge truncation can create a second partial box. Other
                # overlapping crops or the full frame cover these boundaries.
                if mode == 'thorough' and passes > 1 and (
                    (ox > 0 and local[0] <= 2) or (oy > 0 and local[1] <= 2)
                    or (ox+vw < w and local[2] >= vw-2)
                    or (oy+vh < h and local[3] >= vh-2)
                ):
                    continue
                xyxy = [max(0., min(float(w), local[0]+ox)),
                        max(0., min(float(h), local[1]+oy)),
                        max(0., min(float(w), local[2]+ox)),
                        max(0., min(float(h), local[3]+oy))]
                if all(np.isfinite(xyxy)) and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                    candidates.append({'xyxy': xyxy, 'confidence': float(b.conf.item())})
    return {'boxes': merge_boxes(candidates), 'width': w, 'height': h,
            'mode': mode, 'passes': passes,
            'inference_ms': round((time.monotonic()-started)*1000),
            'distance': None}


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html')


@app.get('/api/status')
def status():
    path = Path(os.environ.get('POTHOLE_WEIGHTS', str(ROOT / 'models' / 'best.pt')))
    return {'model_file_present': path.is_file(), 'model_loaded': model is not None}


@app.post('/api/frame')
async def frame(file: UploadFile = File(...), confidence: float = Form(.15),
                imgsz: int = Form(640), mode: str = Form('standard')):
    raw = await file.read(15 * 1024 * 1024 + 1)
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, 'Image exceeds 15 MB.')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, 'Image cannot be decoded.')
    if image.shape[0]*image.shape[1] > 24_000_000:
        raise HTTPException(413, 'Image exceeds 24 megapixels.')
    try:
        return await asyncio.to_thread(detect, image, confidence, imgsz, mode)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


def analyse_video(job_id, path, confidence, imgsz, step, mode='standard'):
    job = jobs[job_id]
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError('Video cannot be decoded. Use an MP4 with H.264 video.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            raise RuntimeError('Video frame rate unavailable.')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job.update(status='analysing', fps=fps, total=total)
        results = []
        i = 0
        while True:
            if job['cancel']:
                job['status'] = 'cancelled'
                return
            ok, image = cap.read()
            if not ok:
                break
            if i % step == 0:
                result = detect(image, confidence, imgsz, mode)
                result.update(frame=i, timestamp=i/fps)
                results.append(result)
            i += 1
            job['done'] = i
        if not results:
            raise RuntimeError('No readable frames found.')
        output = {'fps': fps, 'frame_count': i, 'step': step, 'mode': mode, 'frames': results}
        (DATA / f'{job_id}.json').write_text(json.dumps(output))
        job.update(status='complete', done=i, total=i)
    except Exception as e:
        job.update(status='error', error=str(e))
    finally:
        cap.release()
        path.unlink(missing_ok=True)


@app.post('/api/video')
async def video(file: UploadFile = File(...), confidence: float = Form(.15),
                imgsz: int = Form(640), step: int = Form(1), mode: str = Form('standard')):
    try:
        validate_settings(confidence, imgsz, mode)
        if step not in (1,3,5):
            raise ValueError('Invalid frame sampling setting.')
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if mode == 'thorough':
        step = 1
    job_id = uuid.uuid4().hex
    path = DATA / f'{job_id}.mp4'
    size = 0
    try:
        with path.open('wb') as out:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Video exceeds 300 MB.')
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    jobs[job_id] = {'status':'queued', 'done':0, 'total':0, 'cancel':False}
    executor.submit(analyse_video, job_id, path, confidence, imgsz, step, mode)
    return {'id':job_id}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, 'Analysis job not found.')
    return {k:v for k,v in jobs[job_id].items() if k != 'cancel'}


@app.get('/api/jobs/{job_id}/results')
def results(job_id: str):
    if job_id not in jobs or jobs[job_id]['status'] != 'complete':
        raise HTTPException(404, 'Results not ready.')
    return FileResponse(DATA / f'{job_id}.json', media_type='application/json', filename='detections.json')


@app.delete('/api/jobs/{job_id}')
def cancel(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, 'Analysis job not found.')
    jobs[job_id]['cancel'] = True
    (DATA / f'{job_id}.json').unlink(missing_ok=True)
    return {'cancel_requested':True}
