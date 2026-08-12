#!/usr/bin/env python3
import os
import glob
from typing import List

import pandas as pd
from PIL import Image
from PIL import ImageDraw
import torch
import sam3
import numpy as np 
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)

from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

from sam3.eval.postprocessors import PostProcessImage

# ===================== Path Configuration =====================

# Data root directory (same level as data_index)
DATA_ROOT = ""
DATA_INDEX_DIR = DATA_ROOT

# All text prompts to test
PROMPTS = [
    "lesion",
    "tumor",
    "mass",
    "breast lesion",
    "breast tumor",
    "breast mass",
    "breast cancer",
    "breast nodule",
]

# Output directory for results
RESULT_DIR = "results"

# Number of images per batch (adjust based on VRAM)
BATCH_SIZE = 4

# ===================== Utility Functions =====================


def normalize_rel_path(p: str) -> str:
    """
    Normalize ImagePath:
    - Convert to string
    - Strip leading/trailing whitespace
    - Replace backslashes with forward slashes
    """
    p = str(p).strip()
    if p.lower() in ("nan", "none", ""):
        return ""
    p = p.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def build_abs_path(rel_path: str) -> str:
    """Join relative path to DATA_ROOT"""
    if not rel_path:
        return ""
    if rel_path.startswith("/"):
        return rel_path
    return os.path.join(DATA_ROOT, rel_path)


# ===================== SAM3 Model and Preprocessing Initialization =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TF32 / autocast settings (only effective on CUDA)
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# Load model (using official build_sam3_image_model)
bpe_path = "bpe_simple_vocab_16e6.txt.gz"
print(f"Loading SAM3 model with bpe_path={bpe_path}")
model = build_sam3_image_model(bpe_path=bpe_path).to(device)
model.eval()

# Preprocessing transform (from examples)
transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(
            sizes=1008, max_size=1008, square=True, consistent_transform=False
        ),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Post-processing: output box/mask
postprocessor = PostProcessImage(
    max_dets_per_img=1,  # Keep only top-1 detection per image
    iou_type="segm",
    use_original_sizes_box=True,
    use_original_sizes_mask=True,
    convert_mask_to_rle=False,
    detection_threshold=0.0,  # Set to 0 to ensure top-1 is visible
    to_cpu=False,
)

# ===================== Datapoint Construction Tools =====================

GLOBAL_COUNTER = 1  # Counter for InferenceMetadata ids


def create_empty_datapoint() -> Datapoint:
    """One datapoint corresponds to one image, can have multiple queries"""
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image):
    """Set image for datapoint"""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    """Add a text query to datapoint, return the query id"""
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "Call set_image(datapoint, image) first"

    w, h = datapoint.images[0].size
    query_id = GLOBAL_COUNTER
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # Not used for inference
            is_exhaustive=True,  # Not used for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return query_id


# ===================== Iterate All Images in data_index =====================

def iter_images_from_indices():
    """
    Iterate all xlsx files in data_index,
    yield (PIL image, meta_info_dict) sequentially
    """
    xlsx_files = sorted(glob.glob(os.path.join(DATA_INDEX_DIR, "*.xlsx")))
    if not xlsx_files:
        print(f"[WARN] No xlsx files found in {DATA_INDEX_DIR}")
        return

    print(f"Found {len(xlsx_files)} index tables in {DATA_INDEX_DIR}")

    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[WARN] Failed to read {xlsx_path}: {e}")
            continue

        if "ImagePath" not in df.columns:
            print(f"[WARN] {xlsx_path} missing ImagePath column, skipping")
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


# ===================== Run SAM3 in Batches and Extract Bboxes (Single Prompt) =====================

def run_sam3_on_batch(batch, text_prompt: str):
    """
    batch: List[(PIL.Image, meta_dict)]
    text_prompt: Current text prompt
    Return: list of bbox records (for writing to CSV), max one per image (Top-1)
    """
    if not batch:
        return []

    datapoints: List[Datapoint] = []
    metas = []

    # 1) Construct datapoints and record query_id for each image
    for img, meta in batch:
        dp = create_empty_datapoint()
        set_image(dp, img)
        qid = add_text_prompt(dp, text_prompt)  # Add text prompt for this image

        meta = dict(meta)
        meta["query_id"] = qid
        dp = transform(dp)

        datapoints.append(dp)
        metas.append(meta)

    # 2) Collate and move to device
    collated = collate(datapoints, dict_key="dummy")["dummy"]
    collated = copy_data_to_device(collated, device, non_blocking=True)

    # 3) Inference
    if device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_val, exc_tb): return False
        autocast_ctx = DummyCtx()

    with autocast_ctx, torch.inference_mode():
        output = model(collated)

    # 4) Post-processing: returns {query_id: pred_dict}
    processed_results = postprocessor.process_results(
        output, collated.find_metadatas
    )

    if not isinstance(processed_results, dict):
        print("[WARN] Expected processed_results to be dict, but got", type(processed_results))
        return []

    records = []

    # 5) For each image in batch, get Top-1 detection result by query_id
    for meta in metas:
        qid = meta.get("query_id", None)
        if qid is None:
            continue

        pred = processed_results.get(qid, None)
        if pred is None:
            # This query has no predictions (e.g., no detection at all)
            continue

        scores = pred.get("scores", None)
        boxes = pred.get("boxes", None)

        if scores is None or boxes is None:
            continue
        if scores.numel() == 0 or boxes.numel() == 0:
            continue

        idx = scores.argmax()
        top_score = scores[idx].item()
        top_box = boxes[idx].detach().cpu().tolist()
        if len(top_box) != 4:
            continue

        x1, y1, x2, y2 = top_box

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
            "prompt": text_prompt,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": top_score,
        }
        records.append(rec)

    return records


# ===================== Main Process for One Prompt =====================

def run_for_prompt(text_prompt: str):
    safe_prompt = text_prompt.replace(" ", "_")
    output_csv = os.path.join(
        RESULT_DIR,
        f"sam3_breast_bboxes_{safe_prompt}.csv"
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"\n========== Processing prompt: '{text_prompt}' ==========")
    print(f"[INFO] Results will be saved to: {output_csv}")

    all_records = []
    batch = []
    num_images = 0

    for img, meta in iter_images_from_indices():
        batch.append((img, meta))
        num_images += 1

        if len(batch) >= BATCH_SIZE:
            batch_records = run_sam3_on_batch(batch, text_prompt)
            all_records.extend(batch_records)
            print(
                f"[{text_prompt}] Processed {num_images} images, current bbox count: {len(all_records)}"
            )
            batch = []

    # Process last batch
    if batch:
        batch_records = run_sam3_on_batch(batch, text_prompt)
        all_records.extend(batch_records)
        print(
            f"[{text_prompt}] Last batch complete, total images: {num_images}, total bboxes: {len(all_records)}"
        )

    if not all_records:
        print(
            f"[WARN] prompt='{text_prompt}' got no bbox results, "
            f"consider changing prompt or increasing max_dets_per_img / adjusting threshold."
        )
    else:
        df_out = pd.DataFrame(all_records)
        df_out.to_csv(output_csv, index=False)
        print(f"[OK] prompt='{text_prompt}' wrote {len(all_records)} bbox results to: {output_csv}")


# ===================== Main Function: Loop Over All Prompts =====================

def main():
    for text_prompt in PROMPTS:
        run_for_prompt(text_prompt)


if __name__ == "__main__":
    main()
    # For single image visualization, add a debug function to call separately
