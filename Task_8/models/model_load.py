import csv
import glob
import os
from typing import Optional

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model  # or torch, etc.

from artifacts.length import compute_root_lengths_for_image
from artifacts.postprocessing import reconstruct_mask_for_image
from artifacts.preprocessing import preprocess_image


def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_segmentation_model(model_path):
    """
    Load the pre-trained segmentation model with the custom F1 metric.
    """
    loaded = load_model(model_path, custom_objects={"f1": f1})
    return loaded


def process_directory_streaming(
    image_dir,
    model,
    csv_path,
    patch_size=256,
    pattern="*.png",
    debug_save_dir: Optional[str] = None,
):
    """
    End-to-end batch processor:
    - loads images
    - preprocesses + patchifies
    - runs model inference
    - reconstructs full mask
    - measures root lengths
    - writes CSV rows
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plant Id", "Length (px)", "Tip (px)", "Base (px)"])

        for path in image_paths:
            print(f"Processing {path} ...")

            # 1) PREPROCESS (one image)
            patches, meta, _ = preprocess_image(
                path,
                patch_size=patch_size,
            )

            # 2) PREDICT WITH MODEL
            x = patches.astype("float32")
            pred = model.predict(x)  # e.g. (n_patches, 256, 256, n_classes)
            pred_classes = np.argmax(pred, axis=-1).astype(np.uint8)

            # 3) POSTPROCESS — reconstruct full mask
            _, full_mask = reconstruct_mask_for_image(pred_classes, meta, patch_size)

            # 4) MEASURE ROOT LENGTHS
            rows = compute_root_lengths_for_image(
                full_mask,
                meta,
                debug_save_dir=debug_save_dir,
            )

            # 5) WRITE TO CSV
            for row in rows:
                writer.writerow(
                    [
                        row["plant_id"],
                        row["length_px"],
                        row.get("tip_px"),
                        row.get("base_px"),
                    ]
                )

    print(f"Done. Results saved to {csv_path}")
