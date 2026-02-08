import os
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DEFAULT_HF_HOME = os.getenv("SE_HF_HOME", r"C:\New Volume D\hf-cache")
if DEFAULT_HF_HOME:
    os.makedirs(DEFAULT_HF_HOME, exist_ok=True)
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DEFAULT_HF_HOME, "transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(DEFAULT_HF_HOME, "hub"))

from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy_rao
)
from uncertainty.utils import utils


MODEL_CONFIG = {
    "mistral-7b": {"hf_name": "Mistral-7B-Instruct-v0.1", "t": 1.2, "d": 0.2},
    "falcon-1b": {"hf_name": "falcon-rw-1b", "t": 0.8, "d": 0.15},
    "falcon-7b": {"hf_name": "falcon-7b-instruct", "t": 1.5, "d": 0.3},
    "falcon-13b": {"hf_name": "falcon-13b-instruct", "t": 1.8, "d": 0.25},
    "llama-7b": {"hf_name": "Llama-2-7b-chat", "t": 1.4, "d": 0.2},
    "llama-13b": {"hf_name": "Llama-2-13b-chat", "t": 1.6, "d": 0.2}
}

COLOR_PALETTE = ["blue", "amber", "red", "indigo", "emerald", "rose"]

app = FastAPI(title="Semantic Entropy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    modelId: str = Field(..., min_length=1)
    sampleSize: int = Field(10, ge=2, le=200)


class ClusterResponse(BaseModel):
    id: int
    representative: str
    count: int
    color: str
    variations: List[str]


class AnalyzeResponse(BaseModel):
    seValue: float
    threshold: float
    delta: float
    sampleSize: int
    clusters: List[ClusterResponse]


class AnalyzeAsyncResponse(BaseModel):
    jobId: str
    status: str


class AnalyzeStatusResponse(BaseModel):
    jobId: str
    status: str
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None


_MODEL_CACHE = {}
_ENTAILMENT_MODEL = None
_JOBS: Dict[str, AnalyzeStatusResponse] = {}
_JOBS_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def get_model(model_id: str, max_new_tokens: int):
    model_name = MODEL_CONFIG[model_id]["hf_name"]
    cache_key = f"{model_name}:{max_new_tokens}"
    if cache_key not in _MODEL_CACHE:
        logging.info("Loading model: %s (max_new_tokens=%d)", model_name, max_new_tokens)
        _MODEL_CACHE[cache_key] = HuggingfaceModel(
            model_name=model_name,
            stop_sequences="default",
            max_new_tokens=max_new_tokens
        )
        logging.info("Model ready: %s", model_name)
    return _MODEL_CACHE[cache_key]


def get_entailment_model():
    global _ENTAILMENT_MODEL
    if _ENTAILMENT_MODEL is None:
        logging.info("Loading entailment model...")
        _ENTAILMENT_MODEL = EntailmentDeberta()
        logging.info("Entailment model ready")
    return _ENTAILMENT_MODEL


def build_prompt(question: str) -> str:
    brief = utils.BRIEF_PROMPTS["default"]
    return f"{brief}Question: {question}\nAnswer:".strip()


def build_clusters(responses: List[str], semantic_ids: List[int]) -> List[ClusterResponse]:
    grouped = {}
    for response, sid in zip(responses, semantic_ids):
        grouped.setdefault(sid, []).append(response)

    clusters = []
    for idx, sid in enumerate(sorted(grouped.keys())):
        items = grouped[sid]
        counts = Counter(items)
        representative = counts.most_common(1)[0][0]
        variations = [item for item in items if item != representative]
        unique_variations = list(dict.fromkeys(variations))
        clusters.append(
            ClusterResponse(
                id=idx + 1,
                representative=representative,
                count=len(items),
                color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                variations=unique_variations
            )
        )
    return clusters


def _run_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if req.modelId not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail="Unknown modelId")

    max_new_tokens = int(os.getenv("SE_MAX_NEW_TOKENS", "50"))
    temperature = float(os.getenv("SE_TEMPERATURE", "1.0"))

    model = get_model(req.modelId, max_new_tokens=max_new_tokens)
    entailment_model = get_entailment_model()

    prompt = build_prompt(req.question)
    responses = []
    log_liks = []

    logging.info("Generating %d answers...", req.sampleSize)
    for _ in range(req.sampleSize):
        answer, token_log_likelihoods, _ = model.predict(prompt, temperature=temperature)
        responses.append(answer)
        log_liks.append(token_log_likelihoods)
    logging.info("Finished generation")

    if not responses:
        raise HTTPException(status_code=500, detail="No responses generated")

    log_liks_agg = [float(np.mean(log_lik)) for log_lik in log_liks]
    entailment_inputs = [f"{req.question} {r}" for r in responses]

    semantic_ids = get_semantic_ids(
        entailment_inputs,
        model=entailment_model,
        strict_entailment=True,
        example={"question": req.question}
    )

    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum_normalized")
    se_value = float(predictive_entropy_rao(log_likelihood_per_semantic_id))

    model_cfg = MODEL_CONFIG[req.modelId]
    clusters = build_clusters(responses, semantic_ids)

    return AnalyzeResponse(
        seValue=se_value,
        threshold=model_cfg["t"],
        delta=model_cfg["d"],
        sampleSize=req.sampleSize,
        clusters=clusters
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return _run_analyze(req)


@app.post("/analyze_async", response_model=AnalyzeAsyncResponse)
def analyze_async(req: AnalyzeRequest):
    job_id = uuid.uuid4().hex
    job = AnalyzeStatusResponse(jobId=job_id, status="queued")
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    def _job_runner():
        with _JOBS_LOCK:
            _JOBS[job_id].status = "running"
        try:
            result = _run_analyze(req)
            with _JOBS_LOCK:
                _JOBS[job_id].status = "done"
                _JOBS[job_id].result = result
        except Exception as exc:
            logging.exception("Async analyze failed: %s", exc)
            with _JOBS_LOCK:
                _JOBS[job_id].status = "error"
                _JOBS[job_id].error = str(exc)

    _EXECUTOR.submit(_job_runner)
    return AnalyzeAsyncResponse(jobId=job_id, status="queued")


@app.get("/analyze_status/{job_id}", response_model=AnalyzeStatusResponse)
def analyze_status(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown jobId")
        return job


@app.on_event("startup")
def preload_models():
    if os.getenv("SE_DISABLE_PREWARM", "").strip().lower() in {"1", "true", "yes"}:
        logging.info("Prewarm disabled via SE_DISABLE_PREWARM")
        return
    max_new_tokens = int(os.getenv("SE_MAX_NEW_TOKENS", "50"))
    model_id = os.getenv("SE_WARM_MODEL", "falcon-1b")
    try:
        if model_id in MODEL_CONFIG:
            logging.info("Prewarming model: %s", model_id)
            get_model(model_id, max_new_tokens=max_new_tokens)
        logging.info("Prewarming entailment model")
        get_entailment_model()
        logging.info("Prewarm complete")
    except Exception as exc:
        logging.exception("Prewarm failed: %s", exc)
