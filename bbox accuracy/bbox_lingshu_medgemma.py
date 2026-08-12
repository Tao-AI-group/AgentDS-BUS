#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import json
import math
from typing import Dict, Any, Optional, Tuple, Iterable

import pandas as pd
from PIL import Image

import torch
from transformers import AutoProcessor


BACKEND = "medgemma"   # "lingshu" or "medgemma"

LINGSHU_PATH = (
    ""
)
MEDGEMMA_PATH = (
    ""
)

MODEL_NAME = LINGSHU_PATH if BACKEND == "lingshu" else MEDGEMMA_PATH

DATA_ROOT = ""
DATA_INDEX_DIR = DATA_ROOT

IGNORE_INDEX_BASENAME = "2020-BUSI_index_with_clean_image.xlsx"

OUTPUT_PATH = (
    ""
)
os.makedirs(OUTPUT_PATH, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_PATH, f"{BACKEND}_breast_bboxes_norm2px.csv")

TEST_PROMPT = (
    "You are given a breast ultrasound image. "
    "Locate the lesion and return ONLY the bounding box in strict JSON format, "
    "where the coordinates are normalized between 0 and 1 with respect to the image "
    "width and height, in the form: "
    '{"BBox": [x1, y1, x2, y2]}. '
    "Do NOT add any extra keys or text."
)

LOG_EVERY = 50
MAX_NEW_TOKENS = 128


def load_model_and_processor(backend: str):
    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"[INFO] Loading Lingshu from: {LINGSHU_PATH}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LINGSHU_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(LINGSHU_PATH)
        model.eval()
        return model, processor

    if backend == "medgemma":
        from transformers import AutoModelForImageTextToText
        print(f"[INFO] Loading MedGemma from: {MEDGEMMA_PATH}")
        model = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(MEDGEMMA_PATH)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")

def _decode_generated(processor: AutoProcessor, gen_ids: torch.Tensor) -> str:
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if hasattr(processor, "decode"):
        return processor.decode(gen_ids, skip_special_tokens=True).strip()
    return str(gen_ids)


def normalize_rel_path(p: str) -> str:
    p = str(p).strip()
    if p.lower() in ("nan", "none", ""):
        return ""
    p = p.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p

def build_abs_path(rel_path: str) -> str:
    if not rel_path:
        return ""
    if rel_path.startswith("/"):
        return rel_path
    return os.path.join(DATA_ROOT, rel_path)


def iter_images_from_indices() -> Iterable[Tuple[str, Dict[str, Any]]]:
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        return

    xlsx_files = [p for p in xlsx_files if os.path.basename(p) != IGNORE_INDEX_BASENAME]

    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            continue

        if "ImagePath" not in df.columns:
            continue

        for idx, row in df.iterrows():
            rel_path = normalize_rel_path(row["ImagePath"])
            abs_path = build_abs_path(rel_path)

            if not abs_path or not os.path.exists(abs_path):
                continue

            meta = {
                "index_file": os.path.basename(xlsx_path),
                "row_index": idx,
                "DatasetName": row.get("DatasetName", ""),
                "CaseID": row.get("CaseID", ""),
                "ImageID": row.get("ImageID", ""),
                "BenignMalignant": row.get("BenignMalignant", ""),
                "BIRADS_Category": row.get("BIRADS_Category", ""),
                "ImagePath": abs_path,
                "RelImagePath": rel_path,
            }
            yield abs_path, meta


def _extract_bbox_from_obj(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(obj, dict) or set(obj) != {"BBox"}:
        return None
    bbox = obj["BBox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return None
    if not x1 < x2 or not y1 < y2:
        return None
    return x1, y1, x2, y2

def parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    try:
        return _extract_bbox_from_obj(json.loads(str(text).strip()))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@torch.no_grad()
def run_model_for_image(model, processor: AutoProcessor, image_abs_path: str) -> Dict[str, Any]:
    img = Image.open(image_abs_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": TEST_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
    )

    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    raw_text = _decode_generated(processor, gen_ids)

    bbox = parse_bbox_from_text(raw_text)
    if bbox is None:
        return {
            "raw_text": raw_text,
            "norm_x1": None, "norm_y1": None, "norm_x2": None, "norm_y2": None,
            "x1": None, "y1": None, "x2": None, "y2": None,
        }

    norm_x1, norm_y1, norm_x2, norm_y2 = bbox
    W, H = img.size

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    nx1 = clamp01(norm_x1)
    ny1 = clamp01(norm_y1)
    nx2 = clamp01(norm_x2)
    ny2 = clamp01(norm_y2)

    x1 = nx1 * W
    y1 = ny1 * H
    x2 = nx2 * W
    y2 = ny2 * H

    return {
        "raw_text": raw_text,
        "norm_x1": nx1, "norm_y1": ny1, "norm_x2": nx2, "norm_y2": ny2,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    }

def main():
    print(f"BACKEND: {BACKEND}")
    print(f"Using model path: {MODEL_NAME}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Index dir: {DATA_INDEX_DIR}")
    print(f"Ignore index: {IGNORE_INDEX_BASENAME}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Prompt: {TEST_PROMPT}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] torch device: {device}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, processor = load_model_and_processor(BACKEND)

    records = []
    num_images = 0
    num_with_bbox = 0
    start_all = time.time()

    for image_path, meta in iter_images_from_indices():
        num_images += 1

        try:
            _ = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image: {image_path}, error={e}")
            continue

        try:
            start_inference = time.time()
            result = run_model_for_image(model, processor, image_path)
            inference_time = time.time() - start_inference
            print(f"[DEBUG] Inference completed for {os.path.basename(image_path)} in {inference_time:.2f}s")
        except Exception as e:
            print(f"[ERROR] Inference failed for {image_path}, error={e}")
            continue

        rec = {
            "index_file": meta["index_file"],
            "row_index": meta["row_index"],
            "DatasetName": meta["DatasetName"],
            "CaseID": meta["CaseID"],
            "ImageID": meta["ImageID"],
            "BenignMalignant": meta["BenignMalignant"],
            "BIRADS_Category": meta["BIRADS_Category"],
            "ImagePath": meta["ImagePath"],
            "RelImagePath": meta["RelImagePath"],
            "prompt": TEST_PROMPT,
            "model_output": result["raw_text"],
            "norm_x1": result["norm_x1"],
            "norm_y1": result["norm_y1"],
            "norm_x2": result["norm_x2"],
            "norm_y2": result["norm_y2"],
            "x1": result["x1"],
            "y1": result["y1"],
            "x2": result["x2"],
            "y2": result["y2"],
        }

        if None not in (result["x1"], result["y1"], result["x2"], result["y2"]):
            num_with_bbox += 1

        records.append(rec)

        if num_images % LOG_EVERY == 0:
            elapsed_partial = time.time() - start_all
            rate = num_images / elapsed_partial if elapsed_partial > 0 else 0
            print(
                f"[INFO] Processed {num_images} images, {len(records)} records saved, "
                f"successful bboxes: {num_with_bbox}, "
                f"rate: {rate:.2f} img/s, elapsed: {elapsed_partial:.1f}s"
            )

    elapsed = time.time() - start_all

    if not records:
        print(f"[WARN] No records were generated.")
        return

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Total processed: {num_images} images, {len(records)} records saved, {num_with_bbox} with valid bboxes")
    print(f"[INFO] Total elapsed time: {elapsed:.1f}s")
    print(f"[INFO] Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
