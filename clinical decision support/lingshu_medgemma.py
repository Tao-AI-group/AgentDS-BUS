#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from itertools import islice
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import AutoProcessor

from decision_token_probability import infer_yesno_probability

# ============================================================
# 0) Select model backend to test
#    - "lingshu": Lingshu-32B (Qwen2.5-VL based, logits usually available)
#    - "medgemma": MedGemma-27B-it
#    - "qwen3vl": compatible with original script (optional)
# ============================================================
BACKEND = "medgemma"   # <-- Change here: "lingshu" or "medgemma" or "qwen3vl"

# ============================================================
# 1) Model paths (set to your local paths)
# ============================================================
LINGSHU_PATH = ""
MEDGEMMA_PATH = ""

# If you want to keep original qwen3vl test
QWEN3_VL_PATH = "/path/to/qwen3vl"  # <-- Change to your actual path if using qwen3vl

# ============================================================
# 2) Data paths (follow original qwen3vl baseline logic: iterate all xlsx under DATA_INDEX_DIR)
# ============================================================
DATA_AND_INDEX_DIR = ""
DATA_ROOT = DATA_AND_INDEX_DIR
DATA_INDEX_DIR = DATA_AND_INDEX_DIR

RESULT_DIR = "diagnosis_accuracy/results"
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 4

# ============================================================
# 3) Prompt
# ============================================================
PROMPT_TEXT = (
    "You are an expert breast radiologist. "
    "Based on the given breast ultrasound image, answer the following question.\n"
    "Question: Is the lesion malignant?\n"
    "Answer with exactly one word: yes or no.\n"
    "Answer:"
)

# ============================================================
# 4) Utilities: path/label processing
# ============================================================
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

def parse_label(bm) -> Optional[int]:
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

# ============================================================
# 5) Read ImagePath from all xlsx index tables and yield (PIL image, meta dict)
# ============================================================
def iter_images_from_indices():
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx files found under: {DATA_INDEX_DIR}")
        return

    print(f"[INFO] Found {len(xlsx_files)} xlsx files under {DATA_INDEX_DIR}")

    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[WARN] Failed to read {xlsx_path}: {e}")
            continue

        if "ImagePath" not in df.columns:
            print(f"[WARN] {xlsx_path} missing 'ImagePath' column, skipping.")
            continue

        for idx, row in df.iterrows():
            rel_path = normalize_rel_path(row["ImagePath"])
            abs_path = build_abs_path(rel_path)
            if not abs_path or not os.path.exists(abs_path):
                continue

            try:
                img = Image.open(abs_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to open image {abs_path}: {e}")
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

# ============================================================
# 9) Load model
# ============================================================
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

    if backend == "qwen3vl":
        # If you need to run original qwen3vl
        from transformers import Qwen3VLForConditionalGeneration
        print(f"[INFO] Loading Qwen3-VL from: {QWEN3_VL_PATH}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN3_VL_PATH,
            dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")

# ============================================================
# 10) Main process: inference + save CSV + output metrics
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, processor = load_model_and_processor(BACKEND)

    out_csv = os.path.join(RESULT_DIR, f"{BACKEND}_breast_yesno_baseline.csv")
    print(f"[INFO] Output CSV: {out_csv}")

    records: List[Dict] = []
    y_true: List[int] = []
    y_score: List[float] = []

    n_img = 0
    n_ok = 0
    n_fail = 0

    image_iter = iter(iter_images_from_indices())
    while batch := list(islice(image_iter, BATCH_SIZE)):
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

            # Only enter AUROC computation if gt is valid and p_yes is not NaN
            if gt is not None and not (isinstance(p_yes, float) and np.isnan(p_yes)):
                y_true.append(int(gt))
                y_score.append(float(p_yes))

            if n_img % 50 == 0:
                print(f"[INFO] processed={n_img} ok={n_ok} fail={n_fail} valid_labels={len(y_true)}")

    # save CSV
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(df)} rows to {out_csv}")
    print(f"[INFO] total={n_img} ok={n_ok} fail={n_fail} valid_labels={len(y_true)}")

    # ========== Metrics ==========
    # 1) HARD Accuracy (based on pred_yesno) - can be computed even with medgemma fallback
    df_valid_hard = df[df["gt_label"].notna() & df["pred_yesno"].isin(["yes", "no"])].copy()
    if len(df_valid_hard) > 0:
        y_true_hard = df_valid_hard["gt_label"].astype(int).to_numpy()
        y_pred_hard = (df_valid_hard["pred_yesno"] == "yes").astype(int).to_numpy()
        acc_hard = float((y_true_hard == y_pred_hard).mean())
        print(f"[METRIC] HARD Accuracy  N={len(y_true_hard)}  Acc={acc_hard:.4f}")
    else:
        print("[WARN] No valid rows for HARD Accuracy.")

    # 2) PROB AUROC (requires p_yes)
    if len(y_true) >= 2 and len(set(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            y_true_np = np.array(y_true, dtype=np.int64)
            y_score_np = np.array(y_score, dtype=np.float32)
            auroc = float(roc_auc_score(y_true_np, y_score_np))
            print(f"[METRIC] PROB AUROC (requires logits)  N={len(y_true_np)}  AUROC={auroc:.4f}")
        except Exception as e:
            print(f"[WARN] roc_auc_score failed: {e}")
    else:
        print("[INFO] PROB AUROC skipped (insufficient probability labels or single class).")

if __name__ == "__main__":
    main()
