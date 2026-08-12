#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from decision_token_probability import infer_yesno_probability



QWEN3_VL_PATH = ""
DATA_AND_INDEX_DIR = ""

DATA_ROOT = DATA_AND_INDEX_DIR
DATA_INDEX_DIR = DATA_AND_INDEX_DIR

RESULT_DIR = "diagnosis_accuracy/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_CSV = os.path.join(RESULT_DIR, "qwen3vl_breast_stepbystep_overlay_bbox.csv")

BATCH_SIZE = 4

# ===================== Step 1: ROI localization prompt (output bbox JSON, normalized coordinates) =====================

PROMPT_ROI = (
    "You are given a breast ultrasound image.\n"
    "Locate the lesion and return ONLY the bounding box in strict JSON format, "
    "where the coordinates are normalized between 0 and 1 with respect to the image "
    "width and height, in the form:\n"
    "{\"BBox\": [x1, y1, x2, y2]}.\n"
    "Do NOT add any extra keys or text."
)

# ===================== Step 2: Diagnosis prompt (based on "full image with bounding box" for yes/no) =====================

PROMPT_DIAG = (
    "You are an expert breast radiologist.\n"
    "A red bounding box highlights the lesion of interest in the ultrasound image.\n"
    "Question: Is the lesion malignant?\n"
    "Answer with exactly one word: yes or no.\n"
    "Answer:"
)

# ===================== Utility functions: path handling =====================

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

# ===================== Utility functions: label parsing =====================

def parse_label(bm) -> Optional[int]:
    """
    BenignMalignant -> 0/1
    malignant -> 1, benign -> 0
    Returns None if parsing fails (sample not counted in metrics, but still written to CSV)
    """
    if bm is None:
        return None
    s = str(bm).strip().lower()
    if s in ("1", "malignant", "m", "pos", "positive"):
        return 1
    if s in ("0", "benign", "b", "neg", "negative"):
        return 0
    if "malig" in s:
        return 1
    if "benign" in s:
        return 0
    return None

# ===================== Iterate over all images in data_index =====================

def iter_images_from_indices():
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx files found in {DATA_INDEX_DIR}")
        return

    print(f"[INFO] Found {len(xlsx_files)} index files in {DATA_INDEX_DIR}")

    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[WARN] Failed to read {xlsx_path}: {e}")
            continue

        if "ImagePath" not in df.columns:
            print(f"[WARN] {xlsx_path} missing ImagePath column, skipping this file")
            continue

        for idx, row in df.iterrows():
            rel_path = normalize_rel_path(row["ImagePath"])
            abs_path = build_abs_path(rel_path)

            if not abs_path or not os.path.exists(abs_path):
                continue

            try:
                img = Image.open(abs_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to open image: {abs_path}, err={e}")
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
            yield img, meta

# ===================== Step 1: Generate bbox JSON (normalized coordinates) =====================

def _extract_first_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        value = json.loads(str(text).strip())
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

def _sanitize_bbox_norm(b: dict) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(b, dict) or set(b) != {"BBox"}:
        return None
    values = b["BBox"]
    if not isinstance(values, list) or len(values) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return None
    if not x1 < x2 or not y1 < y2:
        return None
    return x1, y1, x2, y2

@torch.no_grad()
def generate_roi_bbox_norm(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    max_new_tokens: int = 64,
) -> Tuple[Optional[Tuple[float,float,float,float]], str, str]:
    """
    Returns:
      bbox_norm: (x1,y1,x2,y2) in [0,1] or None
      roi_raw: model raw output (for debug)
      status: "ok"/"parse_fail"/"exception:..."
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_ROI},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0, prompt_len:]
    roi_raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    obj = _extract_first_json(roi_raw)
    if obj is None:
        return None, roi_raw, "parse_fail"

    bbox = _sanitize_bbox_norm(obj)
    if bbox is None:
        return None, roi_raw, "parse_fail"

    return bbox, roi_raw, "ok"

# ===================== Plan 1: Draw bounding box on full image (preserve context) =====================

def draw_bbox_on_image(img: Image.Image, bbox_norm: Tuple[float,float,float,float], width: int = 6) -> Image.Image:
    out = img.copy()
    W, H = out.size
    x1n, y1n, x2n, y2n = bbox_norm

    x1 = int(round(x1n * W)); y1 = int(round(y1n * H))
    x2 = int(round(x2n * W)); y2 = int(round(y2n * H))

    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))

    draw = ImageDraw.Draw(out)
    for t in range(width):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(255, 0, 0))
    return out

# ===================== Main function =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[INFO] Loading Qwen3-VL from: {QWEN3_VL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_PATH,
        dtype="auto",
        device_map="auto",
        # If environment supports flash-attn2, try:
        # attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)

    print(f"[INFO] Output CSV: {OUT_CSV}")

    records: List[Dict] = []
    y_true: List[int] = []
    y_score: List[float] = []

    batch: List[Tuple[Image.Image, Dict]] = []
    n_img = 0
    n_diag_ok = 0
    n_diag_fail = 0
    n_roi_ok = 0
    n_roi_fail = 0

    for img, meta in iter_images_from_indices():
        batch.append((img, meta))
        if len(batch) < BATCH_SIZE:
            continue

        for img1, meta1 in batch:
            n_img += 1
            gt = parse_label(meta1.get("BenignMalignant", None))

            # ===== Step 1: ROI bbox =====
            try:
                bbox_norm, roi_raw, roi_status = generate_roi_bbox_norm(model, processor, img1)
            except Exception as e:
                bbox_norm, roi_raw, roi_status = None, "", f"exception:{e}"

            if roi_status == "ok" and bbox_norm is not None:
                n_roi_ok += 1
                diag_img = draw_bbox_on_image(img1, bbox_norm, width=6)
            else:
                n_roi_fail += 1
                diag_img = img1  # fallback: use original image if no bbox

            # ===== Step 2: Diagnosis on overlay image =====
            try:
                p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                    model, processor, diag_img, PROMPT_DIAG
                )
                n_diag_ok += 1
                err_str = ""
            except Exception as e:
                n_diag_fail += 1
                p_yes, pred, yes_logit, no_logit = np.nan, "", np.nan, np.nan
                err_str = str(e)

            rec = dict(meta1)
            rec.update({
                "prompt_roi": PROMPT_ROI,
                "prompt_diag": PROMPT_DIAG,
                "gt_label": gt,
                "roi_status": roi_status,
                "roi_raw": roi_raw,
                "roi_x1": bbox_norm[0] if bbox_norm else np.nan,
                "roi_y1": bbox_norm[1] if bbox_norm else np.nan,
                "roi_x2": bbox_norm[2] if bbox_norm else np.nan,
                "roi_y2": bbox_norm[3] if bbox_norm else np.nan,
                "p_yes_malignant": p_yes,
                "pred_yesno": pred,
                "yes_token_logit": yes_logit,
                "no_token_logit": no_logit,
                "error": err_str,
            })
            records.append(rec)

            if err_str == "" and gt is not None:
                y_true.append(int(gt))
                y_score.append(float(p_yes))

            if n_img % 50 == 0:
                print(f"[INFO] processed={n_img} diag_ok={n_diag_ok} diag_fail={n_diag_fail} roi_ok={n_roi_ok} roi_fail={n_roi_fail} valid_labels={len(y_true)}")

        batch = []

    # Process last batch
    for img1, meta1 in batch:
        n_img += 1
        gt = parse_label(meta1.get("BenignMalignant", None))

        try:
            bbox_norm, roi_raw, roi_status = generate_roi_bbox_norm(model, processor, img1)
        except Exception as e:
            bbox_norm, roi_raw, roi_status = None, "", f"exception:{e}"

        if roi_status == "ok" and bbox_norm is not None:
            n_roi_ok += 1
            diag_img = draw_bbox_on_image(img1, bbox_norm, width=6)
        else:
            n_roi_fail += 1
            diag_img = img1

        try:
            p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                model, processor, diag_img, PROMPT_DIAG
            )
            n_diag_ok += 1
            err_str = ""
        except Exception as e:
            n_diag_fail += 1
            p_yes, pred, yes_logit, no_logit = np.nan, "", np.nan, np.nan
            err_str = str(e)

        rec = dict(meta1)
        rec.update({
            "prompt_roi": PROMPT_ROI,
            "prompt_diag": PROMPT_DIAG,
            "gt_label": gt,
            "roi_status": roi_status,
            "roi_raw": roi_raw,
            "roi_x1": bbox_norm[0] if bbox_norm else np.nan,
            "roi_y1": bbox_norm[1] if bbox_norm else np.nan,
            "roi_x2": bbox_norm[2] if bbox_norm else np.nan,
            "roi_y2": bbox_norm[3] if bbox_norm else np.nan,
            "p_yes_malignant": p_yes,
            "pred_yesno": pred,
            "yes_token_logit": yes_logit,
            "no_token_logit": no_logit,
            "error": err_str,
        })
        records.append(rec)

        if err_str == "" and gt is not None:
            y_true.append(int(gt))
            y_score.append(float(p_yes))

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {len(df)} rows to {OUT_CSV}")
    print(f"[INFO] total={n_img} diag_ok={n_diag_ok} diag_fail={n_diag_fail} roi_ok={n_roi_ok} roi_fail={n_roi_fail} valid_labels={len(y_true)}")

    # Compute metrics
    if len(y_true) >= 2 and len(set(y_true)) == 2:
        y_true_np = np.array(y_true, dtype=np.int64)
        y_score_np = np.array(y_score, dtype=np.float32)
        y_pred_np = (y_score_np >= 0.5).astype(np.int64)

        acc = float((y_pred_np == y_true_np).mean())

        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true_np, y_score_np))
        except Exception as e:
            print(f"[WARN] roc_auc_score failed: {e}")
            auroc = float("nan")

        print(f"[METRIC] N={len(y_true_np)}  Accuracy={acc:.4f}  AUROC={auroc:.4f}")
    else:
        print("[WARN] Insufficient valid labels or only single class, cannot compute AUROC.")

if __name__ == "__main__":
    main()
