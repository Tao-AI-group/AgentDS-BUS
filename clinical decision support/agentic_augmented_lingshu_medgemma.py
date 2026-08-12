#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
from itertools import islice
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
from transformers import AutoProcessor

from decision_token_probability import infer_yesno_probability
from descriptor_schema import (
    parse_morphology_descriptors,
    parse_ultrasound_descriptors,
)

# ============================================================
# 0) Select backend
#    - "lingshu": Lingshu-32B (Qwen2.5-VL, logits usually available)
#    - "medgemma": MedGemma-27B-it
# ============================================================
BACKEND = "lingshu"   # <-- Change here: "lingshu" or "medgemma"

# ============================================================
# 1) Model paths (adjust to your local paths)
# ============================================================
LINGSHU_PATH = ""
MEDGEMMA_PATH = ""

# ============================================================
# 2) Data and index paths (follow original agentic script logic: iterate xlsx)
# ============================================================
DATA_ROOT = ""
DATA_INDEX_DIR = DATA_ROOT

GT_BBOX_CSV = "gt_bboxes_from_masks.csv"

RESULT_DIR = "diagnosis_accuracy/results_agent"
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_AGENT = os.path.join(RESULT_DIR, f"{BACKEND}_agent_GTbbox_GTmask.csv")

BATCH_SIZE = 4  # IO batching; inference per image
CROP_PAD_FRAC = 0.15  # bbox crop padding fraction

# ============================================================
# 3) Prompts (consistent with original agentic script)
# ============================================================
PROMPT_FEATURES_BBOX = (
    "You are an expert breast radiologist.\n"
    "The image shows a breast ultrasound with a red bounding box highlighting the lesion.\n"
    "Extract ultrasound descriptors that are mainly visible from the original ultrasound appearance.\n"
    "The extracted descriptors include:\n"
    "Echo pattern with possible values anechoic, hypoechoic, isoechoic, hyperechoic, or mixed solid and cystic.\n"
    "Internal structure with possible values homogeneous, or heterogeneous.\n"
    "Posterior features with no posterior features, enhancement, or shadowing.\n"
    "Calcification with possible values macrocalcifications, microcalcifications, or absent.\n\n"
    "Return STRICT JSON only, with no markdown or extra text, using exactly this schema:\n"
    "{\n"
    '  "echo_pattern": "anechoic | hypoechoic | isoechoic | hyperechoic | mixed_solid_and_cystic",\n'
    '  "internal_structure": "homogeneous | heterogeneous",\n'
    '  "posterior_features": "no_posterior_features | enhancement | shadowing",\n'
    '  "calcification": ["macrocalcifications | microcalcifications | absent"]\n'
    "}\n"
    "For calcification, return one or more listed values as a JSON array; use absent only when no calcification is present. "
    "Use only the listed values. Do not output unknown, uncertain, or any value outside the schema."
)

PROMPT_FEATURES_MASK = (
    "You are an expert breast radiologist.\n"
    "The image is a binary lesion mask in which white indicates the lesion and black indicates the background.\n"
    "Extract morphology descriptors that are mainly visible from the lesion shape and margin.\n"
    "The descriptors include lesion shape with possible values round, oval, or irregular.\n"
    "Lesion orientation with possible values parallel, or non-parallel.\n"
    "Lesion margin with possible values circumscribed, indistinct, angular, microlobulated, or spiculated.\n\n"
    "Return STRICT JSON only, with no markdown or extra text, using exactly this schema:\n"
    "{\n"
    '  "shape": "round | oval | irregular",\n'
    '  "orientation": "parallel | non_parallel",\n'
    '  "margin": "circumscribed | indistinct | angular | microlobulated | spiculated"\n'
    "}\n"
    "Choose exactly one listed value for every descriptor. Do not output unknown, uncertain, or any value outside the schema."
)

def build_prompt_diag(feature_bbox: dict, feature_mask: dict, shape_metrics: dict) -> str:
    return (
        "You are an expert breast radiologist.\n"
        "A red bounding box highlights the lesion region.\n"
        "You are provided with quantitative shape metrics derived from the lesion mask, "
        "extracted morphology descriptors from the lesion mask and ultrasound descriptors "
        "from the breast ultrasound image.\n"
        f"Quantitative shape metrics: {json.dumps(shape_metrics, ensure_ascii=False)}\n"
        f"Morphology descriptors: {json.dumps(feature_mask, ensure_ascii=False)}\n"
        f"Ultrasound descriptors: {json.dumps(feature_bbox, ensure_ascii=False)}\n"
        "Use all provided information to assess the malignancy risk of the lesion.\n"
        "Determine whether the lesion is malignant.\n"
        "Answer using exactly one word: yes or no."
    )

# ============================================================
# 4) Utils: path/label processing
# ============================================================
def normalize_path(p: str) -> str:
    if p is None:
        return ""
    p = str(p).strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p.lower() in ("nan", "none"):
        return ""
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

def crop_with_padding(img: Image.Image, bbox_px: Tuple[int,int,int,int], pad_frac: float = 0.15) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = bbox_px
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))
    xa = max(0, x1 - pad_x)
    ya = max(0, y1 - pad_y)
    xb = min(W, x2 + pad_x)
    yb = min(H, y2 + pad_y)
    return img.crop((xa, ya, xb, yb))

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

# ============================================================
# 5) Iterate from xlsx (supplement MaskPath)
# ============================================================
def iter_images_from_indices():
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx files in {DATA_INDEX_DIR}")
        return

    print(f"[INFO] Found {len(xlsx_files)} xlsx index files in {DATA_INDEX_DIR}")
    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[WARN] read xlsx failed: {xlsx_path} err={e}")
            continue

        if "ImagePath" not in df.columns:
            print(f"[WARN] {xlsx_path} missing ImagePath column, skipping")
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

            mask_rel = normalize_path(row.get("MaskPath", ""))
            mask_abs = to_abs_path(mask_rel) if mask_rel else ""

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
                "MaskPath": mask_abs,
                "RelMaskPath": mask_rel,
            }
            yield img, meta

def load_mask_as_binary(mask_path: str, target_size: Tuple[int,int]) -> Optional[np.ndarray]:
    if not mask_path or (not os.path.exists(mask_path)):
        return None
    try:
        m = Image.open(mask_path).convert("L")
        if m.size != target_size:
            m = m.resize(target_size, resample=Image.NEAREST)
        arr = np.array(m, dtype=np.uint8)
        binm = arr > 0
        if binm.sum() < 10:
            return None
        return binm
    except Exception:
        return None

def mask_to_pil(mask_bool: np.ndarray) -> Image.Image:
    arr = (mask_bool.astype(np.uint8) * 255)
    return Image.fromarray(arr, mode="L").convert("RGB")

def compute_shape_metrics(mask_bool: np.ndarray) -> Dict:
    """Return exactly the five shape metrics shown in the manuscript figure."""
    mask = np.asarray(mask_bool, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("shape metrics require a non-empty lesion mask")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    bbox_area = float(bw * bh)

    area = int(mask.sum())
    # Perimeter proxy from a 3x3 square erosion: count lesion pixels removed
    # by erosion, i.e. the pixels belonging to the morphological boundary.
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + height, dx:dx + width]
    perimeter = float(np.count_nonzero(mask & ~eroded))

    circularity = float(4.0 * np.pi * area / (perimeter ** 2))
    aspect_ratio = float(bw / bh)
    extent = float(area / bbox_area)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
    }

def _decode_generated(processor: AutoProcessor, gen_ids: torch.Tensor) -> str:
    # Compatible with different processors: prefer tokenizer.decode
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # Some processors provide decode method
    if hasattr(processor, "decode"):
        return processor.decode(gen_ids, skip_special_tokens=True).strip()
    return str(gen_ids)

# ============================================================
# 7) VLM generate JSON with strict schema validation
# ============================================================
@torch.no_grad()
def vlm_generate_json(
    model,
    processor: AutoProcessor,
    image: Image.Image,
    prompt_text: str,
    schema_parser,
    max_new_tokens: int = 256,
) -> Tuple[dict, str]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
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
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    text = _decode_generated(processor, gen_ids)

    jd = schema_parser(text)
    return jd, text

# ============================================================
# 9) Load models (following provided Lingshu/MedGemma loading patterns)
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

    raise ValueError(f"Unknown backend: {backend}")

# ============================================================
# 10) Main evaluation: GT bbox + GT mask agentic
# ============================================================
def run_eval_agent(gt_bbox_map, model, processor, out_csv):
    records: List[Dict] = []

    n_img = 0
    n_ok = 0
    n_fail = 0
    n_has_mask = 0
    n_no_mask = 0
    n_has_bbox = 0
    n_no_bbox = 0

    image_iter = iter(iter_images_from_indices())
    while batch := list(islice(image_iter, BATCH_SIZE)):
        for img1, meta1 in batch:
            n_img += 1
            gt = parse_label(meta1.get("BenignMalignant", None))

            W, H = img1.size

            # bbox
            bbox_raw = gt_bbox_map.get(meta1["ImagePath"], None)
            bbox_px = clip_bbox_px(*bbox_raw, W=W, H=H) if bbox_raw is not None else None
            if bbox_px is None:
                n_no_bbox += 1
                img_bbox = img1
            else:
                n_has_bbox += 1
                img_bbox = draw_bbox(img1, bbox_px, width=6)

            # mask
            mask_bool = load_mask_as_binary(meta1.get("MaskPath",""), target_size=(W, H))
            if mask_bool is None:
                n_no_mask += 1
                img_mask = None
                shape_metrics = {}
            else:
                n_has_mask += 1
                img_mask = mask_to_pil(mask_bool)
                shape_metrics = compute_shape_metrics(mask_bool)

            try:
                if img_mask is None:
                    raise ValueError("AgentDS requires a lesion mask for shape metrics")
                # 1) bbox-view features
                feat_bbox, raw_bbox = vlm_generate_json(
                    model,
                    processor,
                    img_bbox,
                    PROMPT_FEATURES_BBOX,
                    parse_ultrasound_descriptors,
                )

                # 2) mask-view features
                if img_mask is not None:
                    feat_mask, raw_mask = vlm_generate_json(
                        model,
                        processor,
                        img_mask,
                        PROMPT_FEATURES_MASK,
                        parse_morphology_descriptors,
                    )
                else:
                    feat_mask, raw_mask = {}, ""

                # 3) final diagnosis prompt
                prompt_diag = build_prompt_diag(feat_bbox, feat_mask, shape_metrics)

                # 4) yes/no decision-token logits and two-way softmax
                p_yes, pred, yes_logit, no_logit = infer_yesno_probability(
                    model, processor, img_bbox, prompt_diag
                )

                err = ""
                n_ok += 1

            except Exception as e:
                feat_bbox, raw_bbox = {}, ""
                feat_mask, raw_mask = {}, ""
                prompt_diag = ""
                p_yes, pred, yes_logit, no_logit = np.nan, "", np.nan, np.nan
                err = str(e)
                n_fail += 1

            rec = dict(meta1)
            rec.update({
                "backend": BACKEND,
                "method": "GT_bbox+GT_mask_agent",
                "gt_label": gt,
                "bbox_found": bool(bbox_px is not None),
                "mask_found": bool(mask_bool is not None),
                "p_yes_malignant": p_yes,
                "pred_yesno": pred,
                "yes_token_logit": yes_logit,
                "no_token_logit": no_logit,
                "error": err,

                "feat_bbox_json": json.dumps(feat_bbox, ensure_ascii=False),
                "feat_bbox_raw": raw_bbox,
                "feat_mask_json": json.dumps(feat_mask, ensure_ascii=False),
                "feat_mask_raw": raw_mask,
                "shape_metrics_json": json.dumps(shape_metrics, ensure_ascii=False),
                "prompt_diag": prompt_diag,
            })
            records.append(rec)

            if n_img % 50 == 0:
                print(f"[AGENT] processed={n_img} ok={n_ok} fail={n_fail} bbox_found={n_has_bbox}/{n_no_bbox} mask_found={n_has_mask}/{n_no_mask}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[AGENT] wrote {len(df)} rows -> {out_csv}")
    print(f"[AGENT] total={n_img} ok={n_ok} fail={n_fail} bbox(found/missing)={n_has_bbox}/{n_no_bbox} mask(found/missing)={n_has_mask}/{n_no_mask}")

    # HARD accuracy (computable even with medgemma fallback)
    df_valid_hard = df[df["gt_label"].notna() & df["pred_yesno"].isin(["yes", "no"])].copy()
    if len(df_valid_hard) > 0:
        y_true_hard = df_valid_hard["gt_label"].astype(int).to_numpy()
        y_pred_hard = (df_valid_hard["pred_yesno"] == "yes").astype(int).to_numpy()
        acc_hard = float((y_true_hard == y_pred_hard).mean())
        print(f"[METRIC] HARD Accuracy  N={len(y_true_hard)}  Acc={acc_hard:.4f}")
    else:
        print("[WARN] No valid rows for HARD Accuracy.")

    # PROB AUROC (requires non-NaN p_yes values)
    df_valid_prob = df[df["gt_label"].notna() & df["p_yes_malignant"].notna()].copy()
    df_valid_prob = df_valid_prob[~df_valid_prob["p_yes_malignant"].astype(str).str.contains("nan", case=False)]
    if len(df_valid_prob) > 0:
        y_true = df_valid_prob["gt_label"].astype(int).to_numpy()
        y_score = df_valid_prob["p_yes_malignant"].astype(float).to_numpy()
        if len(np.unique(y_true)) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(y_true, y_score))
                print(f"[METRIC] PROB AUROC (computed from logits)  N={len(y_true)}  AUROC={auroc:.4f}")
            except Exception as e:
                print(f"[WARN] AUROC computation failed: {e}")
        else:
            print("[INFO] PROB AUROC skipped (only single class present).")
    else:
        print("[INFO] PROB AUROC skipped (no valid probability values available).")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, processor = load_model_and_processor(BACKEND)

    print("[INFO] Loading ground truth bounding box map...")
    gt_map = load_gt_bbox_map(GT_BBOX_CSV)
    print(f"[INFO] Loaded {len(gt_map)} ground truth bounding box entries")

    print("\n========== Run: GT bbox + GT mask agent (features + diagnosis) ==========")
    run_eval_agent(gt_map, model, processor, OUT_AGENT)

if __name__ == "__main__":
    main()
