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
from openai import OpenAI

# ===================== 基本配置 =====================

BASE_URL = "http://0.0.0.0:8000/v1"
API_KEY = "EMPTY"

MODEL_NAME = (
    ""
)

DATA_ROOT = ""
DATA_INDEX_DIR = DATA_ROOT

model_name_lower = MODEL_NAME.lower()
if "8b" in model_name_lower:
    OUTPUT_PATH = (
        ""  
    )
elif "235b" in model_name_lower:
    OUTPUT_PATH = (
        ""
    )
else:
    OUTPUT_PATH = (
        ""
    )

os.makedirs(OUTPUT_PATH, exist_ok=True)


OUTPUT_CSV = os.path.join(OUTPUT_PATH, "qwen3vl_breast_bboxes_norm2px.csv")


TEST_PROMPT = (
    "You are given a breast ultrasound image. "
    "Locate the lesion and return ONLY the bounding box in strict JSON format, "
    "where the coordinates are normalized between 0 and 1 with respect to the image "
    "width and height, in the form: "
    '{"BBox": [x1, y1, x2, y2]}. '
    "Do NOT add any extra keys or text."
)

LOG_EVERY = 50



client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=3600,
)




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
        print("[INFO] No xlsx files found in index directory.")
        return
    print(f"[INFO] Found {len(xlsx_files)} xlsx files to process.")

    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
            print(f"[DEBUG] Processing xlsx file: {os.path.basename(xlsx_path)}")
        except Exception as e:
            print(f"[WARNING] Failed to read xlsx file {os.path.basename(xlsx_path)}: {e}")
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

def run_qwen_for_image(image_abs_path: str) -> Dict[str, Any]:

    image_url = "file://" + image_abs_path

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": TEST_PROMPT,
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
    )

    raw_text = response.choices[0].message.content
    if isinstance(raw_text, list):
        raw_text = "".join(str(c) for c in raw_text)
    raw_text = str(raw_text)

    bbox = parse_bbox_from_text(raw_text)

    if bbox is None:
        return {
            "raw_text": raw_text,
            "norm_x1": None,
            "norm_y1": None,
            "norm_x2": None,
            "norm_y2": None,
            "x1": None,
            "y1": None,
            "x2": None,
            "y2": None,
        }

    norm_x1, norm_y1, norm_x2, norm_y2 = bbox

    with Image.open(image_abs_path) as img:
        W, H = img.size  # (width, height)

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
        "norm_x1": nx1,
        "norm_y1": ny1,
        "norm_x2": nx2,
        "norm_y2": ny2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }



def main():
    print(f"Using model: {MODEL_NAME}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Index dir: {DATA_INDEX_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Prompt: {TEST_PROMPT}")
    print("=" * 80)

    records = []
    num_images = 0
    num_with_bbox = 0

    start_all = time.time()

    for image_path, meta in iter_images_from_indices():
        num_images += 1

        try:
            _ = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Failed to open image: {image_path}")
            continue

        try:
            result = run_qwen_for_image(image_path)
        except Exception as e:
            print(f"[ERROR] Failed to process image with Qwen: {image_path}, error: {e}")
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
            print(
                f" {num_images} images processed, {num_with_bbox} with bbox"
            )

    elapsed = time.time() - start_all

    if not records:
        print("[WARNING] No valid records to save.")
        return

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Results saved to: {OUTPUT_CSV}")
    print(f"[INFO] Total images processed: {num_images}, with valid bbox: {num_with_bbox}")
    print(f"[INFO] Total time elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
