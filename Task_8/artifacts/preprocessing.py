def extract_dish(img):

    # height, widh, center of image
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, widh = gray.shape
    elif img.ndim == 2:
        gray = img
        height, widh = img.shape

    # apply Otsu thresholding
    th, output_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(f'Otsu algorithm selected the following threshold: {th}')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_im)

    # the biggest blob (object) on each image is Petri dish
    # -infinity - to ensure any score is better
    best_label, best_score = None, -np.inf

    # in loop center is used since Petri dish is near center
    for lab in range(1, num_labels):
        """
        range(1...) - skip label 0 as it is the background
        stats are in format: [x, y, w, h, area] where:
            x.y - top-left corner
            w,h - width and height
            area - area in pixels
        centroids are in format: [cx, cy]
        """
        x, y, w, h, area = stats[lab]
        cx, cy = centroids[lab]

        # big area, near center, square shape
        area_score = area
        center_score = -np.linalg.norm([cx - widh/2, cy - height/2])
        
        #checking aspect ratio
        aspect = w / h
        square_score = -abs(aspect - 1.0)

        # check scores weights (area, square, center)
        score = (1.0 * area_score + 500.0 * square_score + 1000.0 * center_score)

        if score > best_score:
            best_score, best_label = score, lab
    
    x, y, w, h, area = stats[best_label]

    # ensure square that the Pertri dish is fully inside margin is added
    x_m = max(x , 0)
    y_m = max(y , 0)

    # crop
    side = min(w, h)
    x2 = x + side
    y2 = y + side
    crop = img[y_m:y2, x_m:x2]

    # check crop is square
    assert crop.shape[0] == crop.shape[1], "Crop is not square!"
    bbox = (y_m, y2, x_m, x2)
    return crop, bbox


def padder(image, patch_size=256):
    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding,
                                      left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image, (top_padding, bottom_padding, left_padding, right_padding)


def preprocess_image(image_path, patch_size=256, scaling_factor=1.0):
    """
    Returns patches, meta, dish_padded for a single image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read {image_path}")

    original_shape = img.shape

    # 1) crop to dish
    dish, bbox = extract_dish(img)

    # 2) pad to full patches
    dish_padded, padding = padder(dish, patch_size)

    # 3) optional resize
    if scaling_factor != 1.0:
        dish_padded = cv2.resize(
            dish_padded,
            (0, 0),
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )

    H_p, W_p = dish_padded.shape[:2]

    # 4) patchify
    patches = patchify(dish_padded, (patch_size, patch_size, 3), step=patch_size)
    n_rows, n_cols = patches.shape[:2]
    patches = patches.reshape(-1, patch_size, patch_size, 3)

    meta = {
        "image_path": image_path,
        "stem": os.path.splitext(os.path.basename(image_path))[0],
        "original_shape": original_shape,
        "bbox": bbox,
        "padding": padding,
        "padded_shape": (H_p, W_p),
        "n_rows": n_rows,
        "n_cols": n_cols,
    }

    return patches, meta, dish_padded