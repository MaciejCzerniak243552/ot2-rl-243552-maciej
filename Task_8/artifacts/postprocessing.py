import numpy as np


def unpadder(padded_image, padding):
    top_padding, bottom_padding, left_padding, right_padding = padding

    # remove the exact padding added during padder()
    h_end = None if bottom_padding == 0 else -bottom_padding
    w_end = None if right_padding  == 0 else -right_padding

    unpadded = padded_image[
        top_padding : h_end,
        left_padding : w_end
    ]

    return unpadded


def undo_extract_dish(mask_crop, original_shape, bbox):
    y_m, y2, x_m, x2 = bbox
    H, W = original_shape[:2]

    full_mask = np.zeros((H, W), dtype=mask_crop.dtype)
    full_mask[y_m:y2, x_m:x2] = mask_crop

    return full_mask


def reconstruct_mask_for_image(pred_patches, meta, patch_size=256):
    """
    pred_patches: (n_patches, patch_size, patch_size) or (..., 1)
                  for this particular image only.

    meta: corresponding metadata dict.
    """
    n_rows = meta["n_rows"]
    n_cols = meta["n_cols"]
    padding = meta["padding"]
    original_shape = meta["original_shape"]
    bbox = meta["bbox"]

    if pred_patches.ndim == 4 and pred_patches.shape[-1] == 1:
        pred_patches = pred_patches[..., 0]

    # (n_rows, n_cols, H, W)
    pred_patches = pred_patches.reshape(
        n_rows, n_cols, patch_size, patch_size
    )

    H_p = n_rows * patch_size
    W_p = n_cols * patch_size
    crop_padded = np.zeros((H_p, W_p), dtype=pred_patches.dtype)

    for r in range(n_rows):
        for c in range(n_cols):
            patch = pred_patches[r, c]
            y0 = r * patch_size
            x0 = c * patch_size
            crop_padded[y0:y0+patch_size, x0:x0+patch_size] = patch

    crop_unpadded = unpadder(crop_padded, padding)
    full_mask = undo_extract_dish(crop_unpadded, original_shape, bbox)
    return crop_unpadded, full_mask
