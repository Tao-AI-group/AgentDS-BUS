#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from decision_token_probability import infer_yesno_probability

# ===================== Path Configuration =====================

QWEN3_VL_PATH = ""
DATA_ROOT = ""
DATA_INDEX_DIR = DATA_ROOT

GT_BBOX_CSV = "gt_bboxes_from_masks.csv"
SAM3_BBOX_CSV = "bbox_accuracy/model_results/SAM3_results/sam3_breast_bboxes_breast_mass.csv"

RESULT_DIR = "diagnosis_accuracy/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_GT = os.path.join(RESULT_DIR, "qwen3vl_diag_overlay_GTbbox.csv")
OUT_GT_MASK = os.path.join(RESULT_DIR, "qwen3vl_diag_overlay_GTmask.csv")
OUT_SAM3 = os.path.join(RESULT_DIR, "qwen3vl_diag_overlay_SAM3bbox.csv")

BATCH_SIZE = 4  # IO batching only; inference per image

# ===================== Mutually Exclusive ROI Diagnosis Prompts =====================

PROMPT_DIAG_BBOX = (
    "You are an expert breast radiologist.\n"
    "A red bounding box highlights the lesion of interest in the ultrasound image.\n"
    "Question: Is the lesion malignant?\n"
    "Answer with exactly one word: yes or no.\n"
    "Answer:"
)

PROMPT_DIAG_MASK = (
    "You are an expert breast radiologist.\n"
    "The lesion boundary is outlined in green.\n"
    "Question: Is the lesion malignant?\n"
    "Answer with exactly one word: yes or no.\n"
    "Answer:"
)

# ===================== Utility Functions =====================

def normalize_path(p: str) -> str:
    if p is None:
        return ""
    p = str(p).strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p

def to_abs_path(p: str) -> str:
    p = normalize_path(p)
    if not p:
        return ""
    if p.startswith("/"):
        return p
    return os.path.join(DATA_ROOT, p)

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

def clip_bbox_px(x1, y1, x2, y2, W: int, H: int) -> Optional[Tuple[int,int,int,int]]:
    try:
        x1 = int(round(float(x1))); y1 = int(round(float(y1)))
        x2 = int(round(float(x2))); y2 = int(round(float(y2)))
    except Exception:
        return None

    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)

    xa = max(0, min(W - 1, xa))
    xb = max(0, min(W - 1, xb))
    ya = max(0, min(H - 1, ya))
    yb = max(0, min(H - 1, yb))

    if xb <= xa + 1 or yb <= ya + 1:
        return None
    return xa, ya, xb, yb

def draw_bbox(img: Image.Image, bbox_px: Tuple[int,int,int,int], width: int = 6) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox_px
    for t in range(width):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(255, 0, 0))
    return out

def load_mask_as_binary(mask_path: str, target_size: Tuple[int,int]) -> Optional[np.ndarray]:
    if not mask_path or not os.path.exists(mask_path):
        return None
    try:
        mask = Image.open(mask_path).convert("L")
        if mask.size != target_size:
            mask = mask.resize(target_size, resample=Image.NEAREST)
        binary = np.asarray(mask, dtype=np.uint8) > 0
        return binary if binary.sum() >= 10 else None
    except Exception:
        return None

def draw_green_contour(
    img: Image.Image,
    mask: np.ndarray,
    width: int = 3,
) -> Image.Image:
    """Draw only the lesion contour; this function never adds a BBox."""
    out = img.convert("RGB").copy()
    if mask is None:
        return out
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    eroded = mask_img.filter(ImageFilter.MinFilter(3))
    boundary = np.asarray(mask_img, dtype=np.uint8) > np.asarray(eroded, dtype=np.uint8)
    boundary_img = Image.fromarray(boundary.astype(np.uint8) * 255)
    if width > 1:
        boundary_img = boundary_img.filter(ImageFilter.MaxFilter(width * 2 + 1))
    out.paste(Image.new("RGB", out.size, (0, 255, 0)), mask=boundary_img)
    return out

def build_roi_view(
    img: Image.Image,
    meta: Dict,
    roi_type: str,
    bbox_map: Optional[Dict[str, Tuple[float,float,float,float]]] = None,
) -> Tuple[Image.Image, Optional[Tuple[int,int,int,int]], Optional[np.ndarray]]:
    """Build a BBox-only or mask-contour-only ROI view."""
    if roi_type == "bbox":
        bbox_raw = (bbox_map or {}).get(meta["ImagePath"])
        bbox_px = (
            clip_bbox_px(*bbox_raw, W=img.width, H=img.height)
            if bbox_raw is not None
            else None
        )
        view = draw_bbox(img, bbox_px, width=6) if bbox_px is not None else img.copy()
        return view, bbox_px, None
    if roi_type == "mask":
        mask = load_mask_as_binary(meta.get("MaskPath", ""), target_size=img.size)
        view = draw_green_contour(img, mask) if mask is not None else img.copy()
        return view, None, mask
    raise ValueError(f"unsupported roi_type: {roi_type}")

# ===================== Iterate Images from Index xlsx =====================

def iter_images_from_indices():
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx files in {DATA_INDEX_DIR}")
        return

    print(f"[INFO] Found {len(xlsx_files)} xlsx indices in {DATA_INDEX_DIR}")
    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[WARN] read xlsx failed: {xlsx_path} err={e}")
            continue

        if "ImagePath" not in df.columns:
            print(f"[WARN] {xlsx_path} missing ImagePath, skip")
            continue

        for idx, row in df.iterrows():
            rel = normalize_path(row["ImagePath"])
            abs_path = to_abs_path(rel)
            if not abs_path or not os.path.exists(abs_path):
                continue

            try:
                img = Image.open(abs_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] open image failed: {abs_path} err={e}")
                continue

            meta = {
                "index_file": os.path.basename(xlsx_path),
                "row_index": int(idx),
                "DatasetName": row.get("DatasetName", ""),
                "CaseID": row.get("CaseID", ""),
                "ImageID": row.get("ImageID", ""),
                "BenignMalignant": row.get("BenignMalignant", ""),
                "BIRADS_Category": row.get("BIRADS_Category", ""),
                "ImagePath": abs_path,
                "RelImagePath": rel,
                "MaskPath": to_abs_path(row.get("MaskPath", "")),
            }
            yield img, meta

# ===================== Load Bbox Mapping (GT / SAM3) =====================

def load_gt_bbox_map(csv_path: str) -> Dict[str, Tuple[float,float,float,float]]:
    df = pd.read_csv(csv_path)
    need_cols = ["ImagePath", "gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"[GT] missing column {c} in {csv_path}")

    mp: Dict[str, Tuple[float,float,float,float]] = {}
    for _, r in df.iterrows():
        p_abs = to_abs_path(r["ImagePath"])
        if not p_abs:
            continue
        mp[p_abs] = (r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"])
    return mp

def load_sam3_bbox_map(csv_path: str) -> Dict[str, Tuple[float,float,float,float]]:
    df = pd.read_csv(csv_path)
    need_cols = ["ImagePath", "x1", "y1", "x2", "y2"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"[SAM3] missing column {c} in {csv_path}")

    # If score exists: keep the one with highest score for each image
    has_score = ("score" in df.columns)

    mp: Dict[str, Tuple[float,float,float,float]] = {}
    best_score: Dict[str, float] = {}

    for _, r in df.iterrows():
        p_abs = to_abs_path(r["ImagePath"])
        if not p_abs:
            continue

        sc = float(r["score"]) if has_score and pd.notna(r["score"]) else 0.0
        if (p_abs not in mp) or (sc > best_score.get(p_abs, -1e18)):
            mp[p_abs] = (r["x1"], r["y1"], r["x2"], r["y2"])
            best_score[p_abs] = sc

    return mp

# ===================== Run Evaluation (given bbox map) =====================

def run_eval(
    method_name: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    out_csv: str,
    roi_type: str,
    bbox_map: Optional[Dict[str, Tuple[float,float,float,float]]] = None,
):
    if roi_type not in {"bbox", "mask"}:
        raise ValueError(f"unsupported roi_type: {roi_type}")
    prompt_diag = PROMPT_DIAG_BBOX if roi_type == "bbox" else PROMPT_DIAG_MASK
    records: List[Dict] = []
    y_true: List[int] = []
    y_score: List[float] = []

    n_img = 0
    n_hasbox = 0
    n_nobox = 0
    n_ok = 0
    n_fail = 0

    batch: List[Tuple[Image.Image, Dict]] = []

    for img, meta in iter_images_from_indices():
        batch.append((img, meta))
        if len(batch) < BATCH_SIZE:
            continue

        for img1, meta1 in batch:
            n_img += 1
            gt = parse_label(meta1.get("BenignMalignant", None))

            diag_img, bbox_px, roi_mask = build_roi_view(
                img1, meta1, roi_type=roi_type, bbox_map=bbox_map
            )

            roi_found = bbox_px is not None if roi_type == "bbox" else roi_mask is not None
            if not roi_found:
                n_nobox += 1
            else:
                n_hasbox += 1

            try:
                p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                    model, processor, diag_img, prompt_diag
                )
                n_ok += 1
                err = ""
            except Exception as e:
                p_yes, pred, yes_logit, no_logit = np.nan, "", np.nan, np.nan
                n_fail += 1
                err = str(e)

            rec = dict(meta1)
            rec.update({
                "method": method_name,
                "roi_type": roi_type,
                "prompt_diag": prompt_diag,
                "gt_label": gt,
                "bbox_found": bool(bbox_px is not None),
                "mask_found": bool(roi_mask is not None),
                "bbox_x1": bbox_px[0] if bbox_px else np.nan,
                "bbox_y1": bbox_px[1] if bbox_px else np.nan,
                "bbox_x2": bbox_px[2] if bbox_px else np.nan,
                "bbox_y2": bbox_px[3] if bbox_px else np.nan,
                "p_yes_malignant": p_yes,
                "pred_yesno": pred,
                "yes_token_logit": yes_logit,
                "no_token_logit": no_logit,
                "error": err,
            })
            records.append(rec)

            if err == "" and gt is not None:
                y_true.append(int(gt))
                y_score.append(float(p_yes))

            if n_img % 50 == 0:
                print(f"[{method_name}] processed={n_img} ok={n_ok} fail={n_fail} hasbox={n_hasbox} nobox={n_nobox} valid={len(y_true)}")

        batch = []

    # Process last batch
    for img1, meta1 in batch:
        n_img += 1
        gt = parse_label(meta1.get("BenignMalignant", None))

        diag_img, bbox_px, roi_mask = build_roi_view(
            img1, meta1, roi_type=roi_type, bbox_map=bbox_map
        )

        roi_found = bbox_px is not None if roi_type == "bbox" else roi_mask is not None
        if not roi_found:
            n_nobox += 1
        else:
            n_hasbox += 1

        try:
            p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                model, processor, diag_img, prompt_diag
            )
            n_ok += 1
            err = ""
        except Exception as e:
            p_yes, pred, yes_logit, no_logit = np.nan, "", np.nan, np.nan
            n_fail += 1
            err = str(e)

        rec = dict(meta1)
        rec.update({
            "method": method_name,
            "roi_type": roi_type,
            "prompt_diag": prompt_diag,
            "gt_label": gt,
            "bbox_found": bool(bbox_px is not None),
            "mask_found": bool(roi_mask is not None),
            "bbox_x1": bbox_px[0] if bbox_px else np.nan,
            "bbox_y1": bbox_px[1] if bbox_px else np.nan,
            "bbox_x2": bbox_px[2] if bbox_px else np.nan,
            "bbox_y2": bbox_px[3] if bbox_px else np.nan,
            "p_yes_malignant": p_yes,
            "pred_yesno": pred,
            "yes_token_logit": yes_logit,
            "no_token_logit": no_logit,
            "error": err,
        })
        records.append(rec)

        if err == "" and gt is not None:
            y_true.append(int(gt))
            y_score.append(float(p_yes))

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[{method_name}] wrote {len(df)} rows -> {out_csv}")
    print(f"[{method_name}] total={n_img} ok={n_ok} fail={n_fail} hasbox={n_hasbox} nobox={n_nobox} valid={len(y_true)}")

    if len(y_true) >= 2 and len(set(y_true)) == 2:
        y_true_np = np.array(y_true, dtype=np.int64)
        y_score_np = np.array(y_score, dtype=np.float32)
        y_pred_np = (y_score_np >= 0.5).astype(np.int64)

        acc = float((y_pred_np == y_true_np).mean())

        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true_np, y_score_np))
        except Exception as e:
            print(f"[{method_name}] AUROC failed: {e}")
            auroc = float("nan")

        print(f"[{method_name}] METRIC: N={len(y_true_np)}  Accuracy={acc:.4f}  AUROC={auroc:.4f}")
    else:
        print(f"[{method_name}] WARN: not enough valid labels or single class; cannot compute AUROC.")

# ===================== Main =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[INFO] Loading Qwen3-VL from: {QWEN3_VL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)

    print("[INFO] Loading bounding box maps...")
    gt_map = load_gt_bbox_map(GT_BBOX_CSV)
    sam3_map = load_sam3_bbox_map(SAM3_BBOX_CSV)
    print(f"[INFO] GT map entries: {len(gt_map)}")
    print(f"[INFO] SAM3 map entries: {len(sam3_map)}")

    print("\n========== Run 1: GT tight BBox-only ROI ==========")
    run_eval(
        "GT_bbox_only",
        model,
        processor,
        OUT_GT,
        roi_type="bbox",
        bbox_map=gt_map,
    )

    print("\n========== Run 2: GT mask-contour-only ROI ==========")
    run_eval(
        "GT_mask_contour_only",
        model,
        processor,
        OUT_GT_MASK,
        roi_type="mask",
    )

    print("\n========== Run 3: SAM3 BBox-only ROI ==========")
    run_eval(
        "SAM3_bbox_only",
        model,
        processor,
        OUT_SAM3,
        roi_type="bbox",
        bbox_map=sam3_map,
    )

if __name__ == "__main__":
    main()
