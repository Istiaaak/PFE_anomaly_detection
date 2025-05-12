import os
import io
import time
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from PIL import Image

from utils.utils_app import load_patchcore_model
from data.data import MVTecDataset
from logger import get_logger
from utils.utils_kafka import create_consumer

os.makedirs("logs", exist_ok=True)
logger = get_logger("api", logfile="logs/patchcore.log")

app = FastAPI(title="PatchCore Anomaly Detection API")

class BuildRequest(BaseModel):
    cls: str
    backbone_key: str
    f_coreset: float = 0.1
    eps: float      = 0.9
    k_nn: int       = 3
    use_cache: bool = True

app.state.model          = None
app.state.train_scores   = None
app.state.default_thresh = None

@app.post("/build/")
def build(request: BuildRequest):
    try:
        model, train_scores = load_patchcore_model(
            cls          = request.cls,
            backbone_key = request.backbone_key,
            f_coreset    = request.f_coreset,
            eps          = request.eps,
            k_nn         = request.k_nn,
            use_cache    = request.use_cache
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    default_thresh = float(np.percentile(train_scores, 99))

    app.state.model          = model
    app.state.train_scores   = train_scores
    app.state.default_thresh = default_thresh

    logger.info("build", extra={
        "cls":            request.cls,
        "backbone_key":   request.backbone_key,
        "f_coreset":      request.f_coreset,
        "eps":            request.eps,
        "k_nn":           request.k_nn,
        "use_cache":      request.use_cache,
        "default_thresh": default_thresh
    })

    return {"status": "model built", "default_thresh": default_thresh}


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = None
):
    if app.state.model is None:
        raise HTTPException(400, "Model not built: call /build/ first")

    start_time = time.time()

    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    img  = img.resize((app.state.model.image_size, app.state.model.image_size))

    transform = MVTecDataset(
        "dummy",
        size=app.state.model.image_size,
        vanilla=app.state.model.vanilla
    ).get_datasets()[0].transform

    tensor = transform(img).unsqueeze(0)

    score, amap = app.state.model.predict(tensor)
    score_val   = float(score.item())

    thr = threshold if threshold is not None else app.state.default_thresh

    amap_np   = amap.squeeze().cpu().numpy()
    amap_norm = (amap_np - amap_np.min())/(amap_np.max()-amap_np.min()+1e-8)
    mask_bin  = (amap_norm >= thr).astype(int).tolist()

    duration_ms = int((time.time() - start_time) * 1000)

    logger.info("predict", extra={
        "filename":    file.filename,
        "score":       score_val,
        "threshold":   thr,
        "duration_ms": duration_ms
    })

    return JSONResponse({
        "score":     score_val,
        "threshold": thr,
        "mask":      mask_bin
    })


@app.get("/stream_predict/")
def stream_predict(threshold: Optional[float] = None):
    if app.state.model is None:
        raise HTTPException(400, "Model not built: call /build/ first")

    consumer = create_consumer("patchcore_images", group_id="api_consumer")
    msg      = next(consumer)

    img = Image.open(io.BytesIO(msg.value)).convert("RGB")
    img = img.resize((app.state.model.image_size, app.state.model.image_size))

    transform = MVTecDataset(
        "dummy",
        size=app.state.model.image_size,
        vanilla=app.state.model.vanilla
    ).get_datasets()[0].transform

    tensor = transform(img).unsqueeze(0)

    score, amap = app.state.model.predict(tensor)
    score_val   = float(score.item())

    thr = threshold if threshold is not None else app.state.default_thresh

    amap_np   = amap.squeeze().cpu().numpy()
    amap_norm = (amap_np - amap_np.min())/(amap_np.max()-amap_np.min()+1e-8)
    mask_bin  = (amap_norm >= thr).astype(int).tolist()

    logger.info("stream_predict", extra={
        "score":     score_val,
        "threshold": thr
    })

    return {
        "score":     score_val,
        "threshold": thr,
        "mask":      mask_bin
    }