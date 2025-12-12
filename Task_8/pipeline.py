# pipeline.py
import os
import glob
import csv
import numpy as np
import argparse

from artifacts.preprocessing import preprocess_image
from artifacts.postprocessing import reconstruct_mask_for_image
from artifacts.length import compute_root_lengths_for_image
from models.model_load import model, process_directory_streaming

def parse_args():
    parser = argparse.ArgumentParser(description="Root segmentation & length pipeline")

    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory with input images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1) load your trained model
    model_path = "models/maciej_czerniak_243552_multiclass_unet_model_patchsize256px.h5"   # or .pth, etc.
    model = model(model_path)

    # 2) define input images and output csv
    csv_path = "output.csv"

    # 3) run the full pipeline
    process_directory_streaming(
        image_dir=args.img_dir,
        model=model,
        csv_path=csv_path,
        patch_size=256,
        scaling_factor=1.0,
    )
