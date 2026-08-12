#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import json
from itertools import islice
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration

from decision_token_probability import infer_yesno_probability
from descriptor_schema import (
    parse_morphology_descriptors,
    parse_ultrasound_descriptors,
)

# ===================== Path Configuration =====================

QWEN3_VL_PATH = ""
DATA_ROOT = ""
DATA_INDEX_DIR = DATA_ROOT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SAM3_MASK_ROOT = os.environ.get(
    "SAM3_MASK_ROOT", os.path.join(REPO_ROOT, "bbox accuracy", "SAM3_masks")
)
SAM3_BBOX_FALLBACK_ROOT = os.environ.get(
    "SAM3_BBOX_FALLBACK_ROOT",
    os.path.join(REPO_ROOT, "bbox accuracy", "SAM3_bbox_only_fallback_masks"),
)

VOTE_PROMPTS = ("breast lesion", "tumor", "breast tumor")
VOTE_PAIRS = (
    ("breast lesion", "tumor"),
    ("breast lesion", "breast tumor"),
    ("tumor", "breast tumor"),
)
VOTE_TAU = 0.80

RESULT_DIR = "diagnosis_accuracy/results_agent"
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_AGENT = os.path.join(RESULT_DIR, "agentic_majority_voting.csv")

BATCH_SIZE = 4
CROP_PAD_FRAC = 0.15

# ===================== Prompts =====================

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

# ===================== Utility Functions =====================

def safe_name(x: str) -> str:
    s = str(x).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def build_sam3_mask_path(meta: dict, prompt_text: str) -> str:
    prompt_folder = safe_name(prompt_text)
    fname = (
        f"{safe_name(meta.get('DatasetName',''))}__"
        f"{safe_name(meta.get('CaseID',''))}__"
        f"{safe_name(meta.get('ImageID',''))}__"
        f"{safe_name(meta.get('index_file',''))}__"
        f"row{int(meta.get('row_index',-1))}.png"
    )
    return os.path.join(SAM3_MASK_ROOT, prompt_folder, fname)

def build_sam3_bbox_fallback_mask_path(meta: dict) -> str:
    fname = (
        f"{safe_name(meta.get('DatasetName',''))}__"
        f"{safe_name(meta.get('CaseID',''))}__"
        f"{safe_name(meta.get('ImageID',''))}__"
        f"{safe_name(meta.get('index_file',''))}__"
        f"row{int(meta.get('row_index',-1))}.png"
    )
    return os.path.join(SAM3_BBOX_FALLBACK_ROOT, fname)

def validate_sam3_bbox_fallback_metadata(mask_path: str) -> Tuple[bool, str]:
    metadata_file = f"{mask_path}.metadata.json"
    if not os.path.isfile(metadata_file):
        return False, "metadata_missing"
    try:
        with open(metadata_file, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False, "metadata_invalid"
    if not isinstance(metadata, dict):
        return False, "metadata_invalid"
    if metadata.get("metadata_version") != 1:
        return False, "metadata_version_mismatch"
    if metadata.get("kind") != "gt_tight_bbox_fallback":
        return False, "metadata_kind_mismatch"
    if metadata.get("uses_text_prompt") is not False:
        return False, "text_prompt_provenance"
    if metadata.get("prompt_mode") != "bbox_only":
        return False, "prompt_mode_mismatch"
    if not metadata.get("generation_signature"):
        return False, "generation_signature_missing"
    return True, "compatible"

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("nan")
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + 1e-12))

def select_mask_with_majority_rule(
    img_size: Tuple[int,int],
    meta: dict,
    tau: float = VOTE_TAU,
) -> Tuple[Optional[np.ndarray], str, dict]:
    W, H = img_size
    debug = {}

    # Load three SAM3 masks
    m = {}
    for p in VOTE_PROMPTS:
        mp = build_sam3_mask_path(meta, p)
        mb = load_mask_as_binary(mp, target_size=(W, H))
        m[p] = mb
        debug[f"sam3_{safe_name(p)}_path"] = mp
        debug[f"sam3_{safe_name(p)}_ok"] = (mb is not None)

    fallback_path = build_sam3_bbox_fallback_mask_path(meta)
    fallback_metadata_ok, fallback_metadata_reason = (
        validate_sam3_bbox_fallback_metadata(fallback_path)
    )
    fallback_mask = (
        load_mask_as_binary(fallback_path, target_size=(W, H))
        if fallback_metadata_ok
        else None
    )
    debug["sam3_gt_bbox_fallback_path"] = fallback_path
    debug["sam3_gt_bbox_fallback_metadata_ok"] = fallback_metadata_ok
    debug["sam3_gt_bbox_fallback_metadata_reason"] = fallback_metadata_reason
    debug["sam3_gt_bbox_fallback_ok"] = fallback_mask is not None

    def use_sam3_bbox_fallback(reason: str) -> Tuple[np.ndarray, str, dict]:
        if fallback_mask is None:
            raise RuntimeError(
                "SAM3 GT-tight-BBox fallback mask is required but unavailable: "
                f"{fallback_path} ({reason}; provenance={fallback_metadata_reason}). "
                "Run bbox accuracy/generate_sam3_masks.py first."
            )
        debug["consensus"] = False
        debug["reason"] = reason
        return fallback_mask, "sam3_gt_bbox_fallback", debug

    # Missing vote masks trigger SAM3 re-segmentation from the GT tight BBox.
    if any(m[p] is None for p in VOTE_PROMPTS):
        return use_sam3_bbox_fallback("missing_one_or_more_sam3_masks")

    # Compute the three pairwise IoUs among the breast lesion, tumor, and
    # breast tumor SAM3 masks, then use their minimum as the agreement score.
    pairwise_ious = {
        f"{safe_name(left)}__{safe_name(right)}": mask_iou(m[left], m[right])
        for left, right in VOTE_PAIRS
    }
    debug["pairwise_ious"] = pairwise_ious
    agree_score = min(pairwise_ious.values())
    debug["agree_score_min_iou"] = agree_score

    if agree_score >= tau:
        debug["consensus"] = True
        debug["reason"] = f"min_pair_iou>=tau({tau})"
        return m["breast lesion"], "sam3_breast_lesion", debug

    # Low consensus also uses the SAM3 mask generated from the GT tight BBox.
    return use_sam3_bbox_fallback(f"min_pair_iou<{tau}")

def mask_to_bbox_px(mask_bool: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    if mask_bool is None:
        return None
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

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

def iter_images_from_indices():
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx in {DATA_INDEX_DIR}")
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
        m = Image.open(mask_path)
        m = m.convert("L")
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

# ===================== VLM Generation and Strict JSON Validation =====================

@torch.no_grad()
def vlm_generate_json(
    model: Qwen3VLForConditionalGeneration,
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
    inputs = inputs.to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    jd = schema_parser(text)
    return jd, text

# ===================== Main Evaluation =====================

def run_eval_agent(model, processor, out_csv):
    records: List[Dict] = []
    y_true: List[int] = []
    y_score: List[float] = []

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

            # Consensus uses the breast-lesion SAM3 mask. Missing/low-consensus
            # votes use a separate SAM3 mask generated from the GT tight BBox.
            try:
                chosen_mask, mask_source, vote_debug = select_mask_with_majority_rule(
                    img_size=(W, H),
                    meta=meta1,
                    tau=VOTE_TAU,
                )
            except Exception as e:
                rec = dict(meta1)
                rec.update({
                    "method": "SAM3_vote_or_SAM3_GTBBox_fallback",
                    "gt_label": gt,
                    "bbox_found": False,
                    "mask_found": False,
                    "p_yes_malignant": np.nan,
                    "pred_yesno": "",
                    "yes_token_logit": np.nan,
                    "no_token_logit": np.nan,
                    "error": str(e),
                    "mask_source": "",
                    "vote_consensus": False,
                    "vote_reason": "fallback_unavailable",
                    "vote_agree_score_min_iou": np.nan,
                    "vote_debug_json": "",
                })
                records.append(rec)
                n_fail += 1
                print(f"[ERROR] {meta1.get('CaseID')} {meta1.get('ImageID')}: {e}")
                continue

            # Generate bbox from chosen mask
            bbox_px = mask_to_bbox_px(chosen_mask)
            
            img_bbox = img1
            if bbox_px is None:
                n_no_bbox += 1
            else:
                n_has_bbox += 1
                img_bbox = draw_bbox(img1, bbox_px, width=6)

            # Mask-view & shape metrics
            if chosen_mask is None:
                n_no_mask += 1
                img_mask = None
                shape_metrics = {}
            else:
                n_has_mask += 1
                img_mask = mask_to_pil(chosen_mask)
                shape_metrics = compute_shape_metrics(chosen_mask)

            try:
                if img_mask is None:
                    raise ValueError("AgentDS requires a lesion mask for shape metrics")
                # Extract bbox-view features
                feat_bbox, raw_bbox = vlm_generate_json(
                    model,
                    processor,
                    img_bbox,
                    PROMPT_FEATURES_BBOX,
                    parse_ultrasound_descriptors,
                )

                # Extract mask-view features
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

                # Build final diagnosis prompt
                prompt_diag = build_prompt_diag(feat_bbox, feat_mask, shape_metrics)

                # Infer yes/no probability
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
                print(f"[ERROR] {meta1.get('CaseID')} {meta1.get('ImageID')}: {err}")

            rec = dict(meta1)
            rec.update({
                "method": "SAM3_vote_or_SAM3_GTBBox_fallback",
                "gt_label": gt,
                "bbox_found": bool(bbox_px is not None),
                "mask_found": bool(chosen_mask is not None),
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
                "mask_source": mask_source,
                "vote_consensus": vote_debug.get("consensus", False),
                "vote_reason": vote_debug.get("reason", ""),
                "vote_agree_score_min_iou": vote_debug.get("agree_score_min_iou", np.nan),
                "vote_debug_json": json.dumps(vote_debug, ensure_ascii=False),
            })
            records.append(rec)

            if err == "" and gt is not None:
                y_true.append(int(gt))
                y_score.append(float(p_yes))

            if n_img % 10 == 0:
                print(f"[AGENT] processed={n_img} ok={n_ok} fail={n_fail} bbox={n_has_bbox}/{n_no_bbox} mask={n_has_mask}/{n_no_mask} valid={len(y_true)}")

    # 保存
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[AGENT] wrote {len(df)} rows -> {out_csv}")
    print(f"[AGENT] total={n_img} ok={n_ok} fail={n_fail} valid={len(y_true)}")

    if len(y_true) >= 2 and len(set(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(np.array(y_true), np.array(y_score)))
        except Exception as e:
            print(f"[AGENT] AUROC failed: {e}")
            auroc = float("nan")
        print(f"[AGENT] METRIC: N={len(y_true)}  AUROC={auroc:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    print(f"[INFO] Loading Model from: {QWEN3_VL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)

    print("\n========== Run: Agent (SAM3 Voting -> Mask -> BBox -> Diagnosis) ==========")
    run_eval_agent(model, processor, OUT_AGENT)

if __name__ == "__main__":
    main()
