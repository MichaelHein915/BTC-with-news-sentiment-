"""
step3_sentiment.py - FinBERT Sentiment Scoring
================================================
Scores each news headline using ProsusAI/finbert and returns:
  - label:      "positive", "negative", or "neutral"
  - confidence: model's confidence (0-1)
  - numeric:    signed score (-1 to +1): label_sign * confidence

This step is the slowest (runs a transformer model).
Results are cached to `cache/btc_news_raw.csv` so you only score once.
"""

import logging
import os

# Transformers 5.x may start a background thread to convert .bin weights to safetensors
# while the model is already loaded. Running FinBERT inference in parallel with that
# conversion has been observed to segfault on macOS (native race). See modeling_utils.py
# "Thread-auto_conversion". Hugging Face documents DISABLE_SAFETENSORS_CONVERSION for this.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
from tqdm import tqdm
from transformers import pipeline

from config import FINBERT_MODEL

log = logging.getLogger(__name__)

LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def score_finbert(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Run ProsusAI/finbert on a list of headline texts.

    Returns:
        List of dicts with keys: label, confidence, numeric
    """
    log.info(f"Loading FinBERT model: {FINBERT_MODEL}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    classifier = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=-1,           # CPU (-1); set device=0 for GPU
        truncation=True,
        max_length=512,
        model_kwargs={"use_safetensors": False},
    )

    results = []
    log.info(f"Scoring {len(texts)} headlines...")

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT"):
        batch = texts[i : i + batch_size]
        # FinBERT fails on empty strings
        batch = [t if t.strip() else "neutral market update" for t in batch]

        try:
            preds = classifier(batch)
            for pred in preds:
                label = pred["label"].lower()
                conf  = pred["score"]
                results.append({
                    "label":      label,
                    "confidence": conf,
                    "numeric":    LABEL_MAP.get(label, 0.0) * conf,
                })
        except Exception as e:
            log.warning(f"FinBERT batch {i} failed: {e}")
            results.extend([
                {"label": "neutral", "confidence": 0.0, "numeric": 0.0}
            ] * len(batch))

    return results
