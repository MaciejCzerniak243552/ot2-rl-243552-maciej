import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from artifacts.length import compute_root_lengths_for_image
from artifacts.postprocessing import reconstruct_mask_for_image
from artifacts.preprocessing import preprocess_image
from controllers import (
    WorkspaceBounds,
    load_pid_controller,
    load_rl_controller,
    pixels_to_workspace,
    validate_workspace,
)
from models.model_load import load_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline: preprocess -> infer -> postprocess -> CSV + controller hooks"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory with input images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/maciej_czerniak_243552_multiclass_unet_model_patchsize256px.h5",
        help="Path to the pre-trained segmentation model",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join("Task_8", "artifacts", "submission.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size expected by the model",
    )
    parser.add_argument(
        "--pixel_to_mm",
        type=float,
        default=1.0,
        help="Conversion factor from pixels to target units (e.g., mm) for leaderboard format",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for input images",
    )
    parser.add_argument(
        "--debug_save_dir",
        type=str,
        default=None,
        help="Optional directory to save per-plant mask crops for debugging",
    )
    return parser.parse_args()


def validate_submission_rows(rows: List[Dict]) -> None:
    """
    Minimal format validation before saving.
    """
    for r in rows:
        if "Plant ID" not in r or "Length (px)" not in r:
            raise ValueError("Submission row missing required columns: Plant ID / Length (px)")


def save_submission(rows: List[Dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    validate_submission_rows(rows)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Plant ID", "Length (px)"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[Submission] Saved {len(rows)} rows -> {csv_path}")


def dispatch_to_controllers(
    measurements: List[Dict],
    pixel_to_mm: float = 1.0,
    workspace_bounds: WorkspaceBounds = WorkspaceBounds(
        -0.1871, 0.2531, -0.1706, 0.2195, 0.1694, 0.2896, z_default=0.1694
    ),
) -> List[Dict]:
    """
    Convert pixel coordinates to workspace, validate reachability,
    and send movement commands to PID controller (RL controller kept as TODO).
    Returns a log of controller actions for reproducibility.
    """
    pid = load_pid_controller()
    rl = load_rl_controller()  # TODO: implement RL controller behavior

    calibration = {
        "scale": (pixel_to_mm, pixel_to_mm),
        "offset": (0.0, 0.0),
        "z_default": workspace_bounds.z_default,
    }
    logs = []
    for row in measurements:
        tip_px = row.get("tip_px")
        if tip_px is None:
            logs.append({"plant_id": row.get("plant_id"), "status": "skipped", "reason": "no tip coordinate"})
            continue

        workspace_pt = pixels_to_workspace(tip_px, calibration)
        reachable = validate_workspace(workspace_pt, workspace_bounds)
        log_entry = {
            "plant_id": row.get("plant_id"),
            "tip_px": tip_px,
            "workspace_target": workspace_pt,
            "reachable": reachable,
        }
        if not reachable:
            log_entry["status"] = "skipped_out_of_bounds"
            logs.append(log_entry)
            continue

        # PID controller execution
        pid_result = pid.move_to(workspace_pt)
        pid_inoc = pid.inoculate()
        log_entry["pid_move"] = pid_result
        log_entry["pid_inoculate"] = pid_inoc

        # RL controller placeholder (not yet implemented)
        log_entry["rl_move"] = {"status": "todo", "note": "RL controller integration pending"}
        log_entry["rl_inoculate"] = {"status": "todo"}

        logs.append(log_entry)
    return logs


def run_pipeline(
    img_dir: str,
    model_path: str,
    output_csv: str,
    patch_size: int,
    pixel_to_mm: float,
    pattern: str,
    debug_save_dir: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    End-to-end execution: load images, segment roots, measure primary lengths, save a submission, and call the controller interface with detected tip coordinates.
    Returns (submission_rows, controller_logs).
    """
    seg_model = load_segmentation_model(model_path)

    submission_rows: List[Dict] = []
    controller_measurements: List[Dict] = []

    image_paths = sorted(glob.glob(os.path.join(img_dir, pattern)))

    for image_path in image_paths:
        if not image_path.lower().endswith(".png"):
            continue
        path_full = image_path if os.path.isabs(image_path) else os.path.join(img_dir, image_path)
        print(f"[Pipeline] Processing {path_full}")

        # Preprocess: dish crop, patchify, predict, reconstruct full mask
        patches, meta, _ = preprocess_image(
            path_full,
            patch_size=patch_size,
        )
        preds = seg_model.predict(patches.astype("float32"))
        pred_classes = np.argmax(preds, axis=-1).astype(np.uint8)
        _, full_mask = reconstruct_mask_for_image(pred_classes, meta, patch_size)

        # Split full mask into plant components, extract RSA, and measure primary root length per plant
        length_rows = compute_root_lengths_for_image(
            full_mask,
            meta,
            debug_save_dir=debug_save_dir,
        )

        # Convert/normalize + store lengths for submission and controller use
        for r in length_rows:
            length_unit = r["length_px"] * pixel_to_mm
            submission_rows.append(
                {"Plant ID": r["plant_id"], "Length (px)": length_unit}
            )
            controller_measurements.append(r)

    # Validate and save submission
    save_submission(submission_rows, output_csv)

    # Controller hooks (RL kept as TODO per instructions)
    controller_logs = dispatch_to_controllers(
        controller_measurements,
        pixel_to_mm=pixel_to_mm,
        workspace_bounds=WorkspaceBounds(
            -0.1871, 0.2531, -0.1706, 0.2195, 0.1694, 0.2896, z_default=0.1694
        ),
    )

    return submission_rows, controller_logs


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        img_dir=args.img_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        patch_size=args.patch_size,
        pixel_to_mm=args.pixel_to_mm,
        pattern=args.pattern,
        debug_save_dir=args.debug_save_dir,
    )
