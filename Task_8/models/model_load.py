# models/final_model.py
from tensorflow.keras.models import load_model  # or torch, etc.

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model(model_path):
    model = load_model(model_path, custom_objects={"f1": f1})
    return model


def process_directory_streaming(
    image_dir,
    model,
    csv_path,
    patch_size=256,
    scaling_factor=1.0,
    pattern="*.png",
):
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plant Id", "Length"])  # adjust later

        for path in image_paths:
            print(f"Processing {path} ...")

            # 1) PREPROCESS (one image)
            patches, meta, dish_padded = preprocess_image(
                path,
                patch_size=patch_size,
                scaling_factor=scaling_factor,
            )

            # 2) PREDICT WITH MODEL
            x = patches.astype("float32")      # adapt to your model
            pred = model.predict(x)                    # e.g. (n_patches, 256, 256, 1)
            pred_classes = np.argmax(pred, axis=-1).astype(np.uint8)

            # 3) POSTPROCESS – reconstruct full mask
            full_mask = reconstruct_mask(pred_classes, meta, patch_size)

            # 4) MEASURE ROOT LENGTHS (later)
            rows = compute_root_lengths(full_mask, meta)

            # 5) WRITE TO CSV
            for row in rows:
                writer.writerow([row["plant_id"], row["length_px"]])

    print(f"Done. Results saved to {csv_path}")