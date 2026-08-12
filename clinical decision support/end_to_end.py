#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from decision_token_probability import infer_yesno_probability

# ===================== Path Configuration =====================

QWEN3_VL_PATH = ""
DATA_AND_INDEX_DIR = ""

DATA_ROOT = DATA_AND_INDEX_DIR
DATA_INDEX_DIR = DATA_AND_INDEX_DIR

RESULT_DIR = "results_qwen3vl_baseline"
os.makedirs(RESULT_DIR, exist_ok=True)

# Number of images to read per IO batch (inference still done per-image for stability)
BATCH_SIZE = 4

# ===================== Prompt (End-to-end baseline) =====================

PROMPT_TEXT = (
    "You are an expert breast radiologist. "
    "Based on the given breast ultrasound image, answer the following question.\n"
    "Question: Is the lesion malignant?\n"
    "Answer with exactly one word: yes or no.\n"
    "Answer:"
)

# ===================== Utility Functions: Path Processing =====================

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

# ===================== Utility Functions: Label Parsing =====================

def parse_label(bm) -> Optional[int]:
    """
    BenignMalignant -> 0/1
    malignant -> 1, benign -> 0
    Returns None if unable to parse (sample excluded from metrics but written to CSV)
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

# ===================== Iterate All Images in data_index =====================

def iter_images_from_indices():
    """
    Iterate all xlsx files in data_index
    yield (PIL image, meta dict)
    """
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx files found in {DATA_INDEX_DIR}")
        return

    print(f"[INFO] Found {len(xlsx_files)} index tables in {DATA_INDEX_DIR}")

    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[WARN] Failed to read {xlsx_path}: {e}")
            continue

        if "ImagePath" not in df.columns:
            print(f"[WARN] {xlsx_path} missing ImagePath column, skipping this table")
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

# ===================== Main Function: Inference + Metrics =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device = {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[INFO] Loading Qwen3-VL from: {QWEN3_VL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_PATH,
        dtype="auto",
        device_map="auto",
        # If environment supports flash-attn2, you can try:
        # attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)

    out_csv = os.path.join(RESULT_DIR, "diagnosis_accuracy/results/qwen3vl_breast_yesno_baseline.csv")
    print(f"[INFO] Output CSV: {out_csv}")

    records: List[Dict] = []
    y_true: List[int] = []
    y_score: List[float] = []

    batch: List[Tuple[Image.Image, Dict]] = []
    n_img = 0
    n_ok = 0
    n_fail = 0

    for img, meta in iter_images_from_indices():
        batch.append((img, meta))
        if len(batch) < BATCH_SIZE:
            continue

        for img1, meta1 in batch:
            n_img += 1
            gt = parse_label(meta1.get("BenignMalignant", None))

            try:
                p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                    model, processor, img1, PROMPT_TEXT
                )
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"[WARN] inference failed @ {meta1.get('ImagePath','')} err={e}")
                rec = dict(meta1)
                rec.update({
                    "prompt": PROMPT_TEXT,
                    "gt_label": gt,
                    "p_yes_malignant": np.nan,
                    "pred_yesno": "",
                    "yes_token_logit": np.nan,
                    "no_token_logit": np.nan,
                    "error": str(e),
                })
                records.append(rec)
                continue

            rec = dict(meta1)
            rec.update({
                "prompt": PROMPT_TEXT,
                "gt_label": gt,
                "p_yes_malignant": p_yes,
                "pred_yesno": pred,
                "yes_token_logit": yes_logit,
                "no_token_logit": no_logit,
                "error": "",
            })
            records.append(rec)

            if gt is not None:
                y_true.append(int(gt))
                y_score.append(float(p_yes))

            if n_img % 50 == 0:
                print(f"[INFO] Processed={n_img} ok={n_ok} fail={n_fail} valid_labels={len(y_true)}")

        batch = []

    # Last batch
    for img1, meta1 in batch:
        n_img += 1
        gt = parse_label(meta1.get("BenignMalignant", None))

        try:
            p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                model, processor, img1, PROMPT_TEXT
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] inference failed @ {meta1.get('ImagePath','')} err={e}")
            rec = dict(meta1)
            rec.update({
                "prompt": PROMPT_TEXT,
                "gt_label": gt,
                "p_yes_malignant": np.nan,
                "pred_yesno": "",
                "yes_token_logit": np.nan,
                "no_token_logit": np.nan,
                "error": str(e),
            })
            records.append(rec)
            continue

        rec = dict(meta1)
        rec.update({
            "prompt": PROMPT_TEXT,
            "gt_label": gt,
            "p_yes_malignant": p_yes,
            "pred_yesno": pred,
            "yes_token_logit": yes_logit,
            "no_token_logit": no_logit,
            "error": "",
        })
        records.append(rec)

        if gt is not None:
            y_true.append(int(gt))
            y_score.append(float(p_yes))

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(df)} rows to {out_csv}")
    print(f"[INFO] Total={n_img} ok={n_ok} fail={n_fail} valid_labels={len(y_true)}")

    # Metrics
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
        print("[WARN] Insufficient valid labels or only single class, unable to compute AUROC.")

if __name__ == "__main__":
    main()
