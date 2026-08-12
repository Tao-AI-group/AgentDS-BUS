#!/usr/bin/env python3
"""Generate SAM3 vote masks and GT-tight-BBox fallback masks.

The GT mask is used only to derive an unexpanded tight XYXY box. That box is
then supplied to SAM3 as one positive visual prompt; only the resulting SAM3
mask is saved for downstream AgentDS prediction.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_PROMPTS = [
    "lesion",
    "tumor",
    "mass",
    "breast lesion",
    "breast tumor",
    "breast mass",
    "breast cancer",
    "breast nodule",
]
FALLBACK_METADATA_VERSION = 1
FALLBACK_PROMPT_MODE = "bbox_only"


def normalize_path(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text[2:] if text.startswith("./") else text


def resolve_path(value: object, data_root: Path) -> Path:
    path = Path(normalize_path(value))
    return path if path.is_absolute() else data_root / path


def safe_name(value: object) -> str:
    text = str(value).strip().replace(" ", "_")
    text = re.sub(r"[^\w\-.]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def output_filename(meta: Dict[str, object]) -> str:
    return (
        f"{safe_name(meta.get('DatasetName', ''))}__"
        f"{safe_name(meta.get('CaseID', ''))}__"
        f"{safe_name(meta.get('ImageID', ''))}__"
        f"{safe_name(meta.get('index_file', ''))}__"
        f"row{int(meta.get('row_index', -1))}.png"
    )


def iter_index_rows(index_dir: Path, data_root: Path) -> Iterable[Dict[str, object]]:
    for xlsx_path_text in sorted(glob.glob(str(index_dir / "*.xlsx"))):
        xlsx_path = Path(xlsx_path_text)
        frame = pd.read_excel(xlsx_path)
        if "ImagePath" not in frame or "MaskPath" not in frame:
            print(f"[WARN] skipping {xlsx_path}: ImagePath or MaskPath is missing")
            continue
        for row_index, row in frame.iterrows():
            yield {
                "index_file": xlsx_path.name,
                "row_index": int(row_index),
                "DatasetName": row.get("DatasetName", ""),
                "CaseID": row.get("CaseID", ""),
                "ImageID": row.get("ImageID", ""),
                "ImagePath": str(resolve_path(row.get("ImagePath", ""), data_root)),
                "MaskPath": str(resolve_path(row.get("MaskPath", ""), data_root)),
            }


def load_binary_mask(path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, resample=Image.NEAREST)
    binary = np.asarray(mask, dtype=np.uint8) > 0
    return binary if binary.sum() >= 10 else None


def tight_bbox(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return bbox if bbox[2] > bbox[0] + 1 and bbox[3] > bbox[1] + 1 else None


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(SCRIPT_DIR.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def metadata_path(mask_path: Path) -> Path:
    return mask_path.with_name(f"{mask_path.name}.metadata.json")


def build_fallback_metadata(
    *,
    model_path: str,
    model_revision: str,
    threshold: float,
    mask_threshold: float,
    bbox: Tuple[int, int, int, int],
    commit_sha: str,
    generator_sha256: str,
) -> Dict[str, object]:
    signature_payload = {
        "metadata_version": FALLBACK_METADATA_VERSION,
        "kind": "gt_tight_bbox_fallback",
        "prompt_mode": FALLBACK_PROMPT_MODE,
        "uses_text_prompt": False,
        "model_path": model_path,
        "model_revision": model_revision,
        "generation_parameters": {
            "threshold": float(threshold),
            "mask_threshold": float(mask_threshold),
            "input_boxes_labels": [[1]],
            "selection_rule": "largest_mask_area",
            "input_bbox_xyxy": list(bbox),
        },
        "generator_sha256": generator_sha256,
    }
    signature_text = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return {
        **signature_payload,
        "code_commit_sha": commit_sha,
        "generation_signature": hashlib.sha256(signature_text.encode("utf-8")).hexdigest(),
    }


def write_metadata(mask_path: Path, metadata: Dict[str, object]) -> None:
    path = metadata_path(mask_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def compatible_fallback_exists(
    mask_path: Path, expected_metadata: Dict[str, object]
) -> Tuple[bool, str]:
    if not mask_path.exists():
        return False, "mask_missing"
    path = metadata_path(mask_path)
    if not path.exists():
        return False, "metadata_missing"
    try:
        saved_metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "metadata_invalid"
    if not isinstance(saved_metadata, dict):
        return False, "metadata_invalid"
    if saved_metadata.get("uses_text_prompt") is not False:
        return False, "text_prompt_provenance"
    if saved_metadata.get("prompt_mode") != FALLBACK_PROMPT_MODE:
        return False, "prompt_mode_mismatch"
    if saved_metadata.get("generation_signature") != expected_metadata.get(
        "generation_signature"
    ):
        return False, "generation_signature_mismatch"
    return True, "compatible"


def remove_incompatible_fallback(mask_path: Path) -> None:
    mask_path.unlink(missing_ok=True)
    metadata_path(mask_path).unlink(missing_ok=True)


def status_provenance(metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "code_commit_sha": metadata["code_commit_sha"],
        "generator_sha256": metadata["generator_sha256"],
        "uses_text_prompt": metadata["uses_text_prompt"],
        "prompt_mode": metadata["prompt_mode"],
        "model_path": metadata["model_path"],
        "model_revision": metadata["model_revision"],
        "generation_parameters": json.dumps(
            metadata["generation_parameters"], sort_keys=True
        ),
        "generation_signature": metadata["generation_signature"],
    }


def masks_to_numpy(masks) -> Optional[np.ndarray]:
    if masks is None:
        return None
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return masks if masks.ndim == 3 and masks.shape[0] else None


@torch.no_grad()
def segment_with_text(model, processor, device, image, text, threshold, mask_threshold):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = masks_to_numpy(result.get("masks"))
    if masks is None:
        return None, {"status": "no_masks"}
    scores = result.get("scores")
    if scores is None:
        index = int(np.argmax((masks > mask_threshold).reshape(len(masks), -1).sum(axis=1)))
        selected_score = np.nan
    else:
        scores = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        index = int(np.argmax(scores))
        selected_score = float(scores[index])
    return masks[index] > mask_threshold, {
        "status": "ok",
        "selection_rule": "top_score",
        "selected_score": selected_score,
        "n_instances": int(len(masks)),
    }


@torch.no_grad()
def segment_with_bbox(model, processor, device, image, bbox, threshold, mask_threshold):
    inputs = processor(
        images=image,
        input_boxes=[[list(map(float, bbox))]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = masks_to_numpy(result.get("masks"))
    if masks is None:
        return None, {"status": "no_masks", "input_bbox_xyxy": list(bbox)}
    areas = (masks > mask_threshold).reshape(len(masks), -1).sum(axis=1)
    index = int(np.argmax(areas))
    return masks[index] > mask_threshold, {
        "status": "ok",
        "selection_rule": "largest_mask_area",
        "n_instances": int(len(masks)),
        "input_bbox_xyxy": list(bbox),
    }


def write_status(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")))
    parser.add_argument(
        "--data_index_dir",
        type=Path,
        default=Path(os.environ.get("DATA_INDEX_DIR", ".")),
    )
    parser.add_argument(
        "--sam3_model_path",
        default=os.environ.get("SAM3_MODEL_PATH", "facebook/sam3"),
    )
    parser.add_argument(
        "--sam3_model_revision",
        default=os.environ.get("SAM3_MODEL_REVISION"),
    )
    parser.add_argument(
        "--sam3_mask_root",
        type=Path,
        default=Path(os.environ.get("SAM3_MASK_ROOT", SCRIPT_DIR / "SAM3_masks")),
    )
    parser.add_argument(
        "--fallback_mask_root",
        type=Path,
        default=Path(
            os.environ.get(
                "SAM3_BBOX_FALLBACK_ROOT",
                SCRIPT_DIR / "SAM3_bbox_only_fallback_masks",
            )
        ),
    )
    parser.add_argument("--prompts", default=",".join(TEXT_PROMPTS))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    from transformers import Sam3Model, Sam3Processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    model_load_kwargs = {"local_files_only": args.local_files_only}
    if args.sam3_model_revision:
        model_load_kwargs["revision"] = args.sam3_model_revision
    model = Sam3Model.from_pretrained(args.sam3_model_path, **model_load_kwargs).to(device)
    processor = Sam3Processor.from_pretrained(args.sam3_model_path, **model_load_kwargs)
    model.eval()

    resolved_model_revision = (
        getattr(model.config, "_commit_hash", None)
        or args.sam3_model_revision
        or "unspecified"
    )
    commit_sha = code_commit_sha()
    generator_sha256 = sha256_file(Path(__file__).resolve())

    prompts = [text.strip() for text in args.prompts.split(",") if text.strip()]
    rows = list(iter_index_rows(args.data_index_dir, args.data_root))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise RuntimeError(f"no index rows found in {args.data_index_dir}")

    status_rows = []
    status_path = SCRIPT_DIR / "sam3_mask_generation_status.csv"
    for index, meta in enumerate(rows, start=1):
        image_path = Path(str(meta["ImagePath"]))
        mask_path = Path(str(meta["MaskPath"]))
        filename = output_filename(meta)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            status_rows.append({**meta, "kind": "image", "status": "error", "error": str(exc)})
            continue

        for prompt in prompts:
            output_path = args.sam3_mask_root / safe_name(prompt) / filename
            if output_path.exists() and args.resume:
                status_rows.append({**meta, "kind": "text", "prompt": prompt, "status": "skipped", "mask_path": str(output_path), "error": ""})
                continue
            try:
                predicted, debug = segment_with_text(
                    model, processor, device, image, prompt, args.threshold, args.mask_threshold
                )
                if predicted is None:
                    raise RuntimeError(json.dumps(debug))
                save_mask(predicted, output_path)
                status_rows.append({**meta, "kind": "text", "prompt": prompt, "status": "ok", "mask_path": str(output_path), "debug_json": json.dumps(debug), "error": ""})
            except Exception as exc:
                status_rows.append({**meta, "kind": "text", "prompt": prompt, "status": "error", "mask_path": str(output_path), "error": str(exc)})

        fallback_path = args.fallback_mask_root / filename
        expected_metadata = None
        resume_reason = "resume_disabled"
        try:
            gt_mask = load_binary_mask(mask_path, image.size)
            bbox = tight_bbox(gt_mask)
            if bbox is None:
                if args.resume:
                    remove_incompatible_fallback(fallback_path)
                raise ValueError("cannot derive a valid tight BBox from the GT mask")
            expected_metadata = build_fallback_metadata(
                model_path=str(args.sam3_model_path),
                model_revision=str(resolved_model_revision),
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                bbox=bbox,
                commit_sha=commit_sha,
                generator_sha256=generator_sha256,
            )
            can_resume, resume_reason = compatible_fallback_exists(
                fallback_path, expected_metadata
            )
            if args.resume and can_resume:
                status_rows.append(
                    {
                        **meta,
                        "kind": "gt_tight_bbox_fallback",
                        "status": "skipped",
                        "mask_path": str(fallback_path),
                        "resume_validation": resume_reason,
                        **status_provenance(expected_metadata),
                        "error": "",
                    }
                )
            else:
                if args.resume and (
                    fallback_path.exists() or metadata_path(fallback_path).exists()
                ):
                    print(
                        f"[INFO] regenerating incompatible fallback mask: "
                        f"{fallback_path} ({resume_reason})"
                    )
                    remove_incompatible_fallback(fallback_path)
                predicted, debug = segment_with_bbox(
                    model, processor, device, image, bbox, args.threshold, args.mask_threshold
                )
                if predicted is None:
                    raise RuntimeError(json.dumps(debug))
                save_mask(predicted, fallback_path)
                write_metadata(fallback_path, expected_metadata)
                status_rows.append(
                    {
                        **meta,
                        "kind": "gt_tight_bbox_fallback",
                        "status": "ok",
                        "mask_path": str(fallback_path),
                        "input_bbox_xyxy": json.dumps(list(bbox)),
                        "resume_validation": resume_reason,
                        "debug_json": json.dumps(debug),
                        **status_provenance(expected_metadata),
                        "error": "",
                    }
                )
        except Exception as exc:
            remove_incompatible_fallback(fallback_path)
            provenance = (
                status_provenance(expected_metadata) if expected_metadata else {}
            )
            status_rows.append(
                {
                    **meta,
                    "kind": "gt_tight_bbox_fallback",
                    "status": "error",
                    "mask_path": str(fallback_path),
                    "resume_validation": resume_reason,
                    **provenance,
                    "error": str(exc),
                }
            )

        if index % args.print_every == 0:
            print(f"[SAM3] processed {index}/{len(rows)}")
            write_status(status_rows, status_path)

    write_status(status_rows, status_path)
    print(f"[OK] status: {status_path}")


if __name__ == "__main__":
    main()
