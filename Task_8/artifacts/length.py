from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import binary_closing, skeletonize, square
from skan.csr import skeleton_to_csgraph


def _longest_path_length(coords, graph):
    """
    Compute the longest geodesic path length from the topmost node to any tip in the graph.
    Returns (length, base_node, tip_node).
    """
    if graph.number_of_nodes() == 0:
        return 0.0, None, None

    tips = [n for n, d in graph.degree() if d == 1]
    if not tips:
        return 0.0, None, None

    # choose topmost node as base
    base = min(graph.nodes, key=lambda n: (coords[n, 0], coords[n, 1]))

    dist = nx.single_source_dijkstra_path_length(graph, base, weight="weight")
    best_tip = None
    best_len = -1.0
    for tip in tips:
        L = dist.get(tip, -1.0)
        if L > best_len:
            best_len = L
            best_tip = tip

    return float(max(best_len, 0.0)), base, best_tip


def primary_length_px_from_mask(mask, close_k=4, min_area=50):
    """
    Estimate primary root length (px) from a binary plant mask by:
    - closing gaps
    - keeping the largest component
    - skeletonizing
    - measuring longest geodesic path on the skeleton graph
    Returns (length_px, base_xy, tip_xy, graph_info).
    """
    if mask is None:
        return 0.0, None, None, {}

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < min_area:
        return 0.0, None, None, {}

    binary = binary_closing(binary, square(close_k))
    num, labels = cv2.connectedComponents(binary.astype(np.uint8))
    if num <= 1:
        return 0.0, None, None, {}

    counts = np.bincount(labels.ravel())
    main_label = counts[1:].argmax() + 1
    binary = (labels == main_label).astype(np.uint8)

    skel = skeletonize(binary)
    g0, c0 = skeleton_to_csgraph(skel)
    coords = np.column_stack(c0)
    if coords.size == 0 or g0.shape[0] == 0:
        return 0.0, None, None, {}

    G = nx.from_scipy_sparse_array(g0)
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    G = G.subgraph(largest).copy()

    valid_nodes = {n for n in G.nodes if n < coords.shape[0]}
    if not valid_nodes:
        return 0.0, None, None, {}
    G = G.subgraph(valid_nodes).copy()

    length_px, base_node, tip_node = _longest_path_length(coords, G)

    base_xy = None
    tip_xy = None
    if base_node is not None:
        base_xy = (int(coords[base_node, 1]), int(coords[base_node, 0]))
    if tip_node is not None:
        tip_xy = (int(coords[tip_node, 1]), int(coords[tip_node, 0]))

    graph_info = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "tips": len([n for n, d in G.degree() if d == 1]),
        "base_node": base_node,
        "tip_node": tip_node,
    }
    return length_px, base_xy, tip_xy, graph_info


def detect_plants_per_column(
    full_mask,
    columns=5,
    root_area_min=80,
    root_height_min=20,
    root_aspect_min=2.0,
    seed_area_min=20,
    seed_height_max=80,
    pad=20,
    debug_save_dir=None,
    stem="image",
):
    """
    Per-column plant detector inspired by Task 6 fifth iteration.
    Returns a list of dicts with masks and bboxes (y0, y1, x0, x1).
    """
    H, W = full_mask.shape
    cols = np.linspace(0, W, columns + 1, dtype=int)

    root_label = 1
    shoot_seed_labels = {2, 3}
    root_mask = (full_mask == root_label).astype(np.uint8)

    plants = []
    for col_idx in range(columns):
        x0, x1 = cols[col_idx], cols[col_idx + 1]
        col = full_mask[:, x0:x1]

        # try root channel first
        root_col = (col == root_label).astype(np.uint8)
        kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        root_closed = cv2.morphologyEx(root_col, cv2.MORPH_CLOSE, kernel_vert)
        kernel_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        root_closed = cv2.morphologyEx(root_closed, cv2.MORPH_CLOSE, kernel_horz)
        root_clean = cv2.morphologyEx(root_closed, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

        num_r, lab_r, stats_r, _ = cv2.connectedComponentsWithStats(root_clean, connectivity=8)

        best_root = None
        best_root_score = -np.inf
        for lab in range(1, num_r):
            x, y, w, h, area = stats_r[lab]
            if area < root_area_min or h < root_height_min:
                continue
            aspect = h / max(w, 1)
            if aspect < root_aspect_min:
                continue
            score = area
            if score > best_root_score:
                best_root_score = score
                best_root = (x, y, w, h)

        chosen_box = None
        chosen_mask = None
        chosen_class = None

        if best_root is not None:
            x, y, w, h = best_root
            chosen_box = (x0 + x, y, w, h)
            # crop from root mask to keep consistent pixels
            y0 = max(y - pad, 0)
            x0_adj = max(x0 + x - pad, 0)
            y1 = min(y + h + pad, H)
            x1 = min(x0 + x + w + pad, W)
            crop_mask = root_mask[y0:y1, x0_adj:x1]
            chosen_mask = crop_mask
            chosen_class = "root"
        else:
            # fallback to seed/shoot when root not found
            seed_col = np.isin(col, list(shoot_seed_labels)).astype(np.uint8)
            num_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(seed_col, connectivity=8)
            best_seed = None
            best_seed_area = -np.inf
            for lab in range(1, num_s):
                x, y, w, h, area = stats_s[lab]
                if area < seed_area_min:
                    continue
                if h > seed_height_max:
                    continue
                if area > best_seed_area:
                    best_seed_area = area
                    best_seed = (x, y, w, h)
            if best_seed is not None:
                x, y, w, h = best_seed
                chosen_box = (x0 + x, y, w, h)
                y0 = max(y - pad, 0)
                x0_adj = max(x0 + x - pad, 0)
                y1 = min(y + h + pad, H)
                x1 = min(x0 + x + w + pad, W)
                # keep only root channel in the saved mask for consistency
                crop_mask = root_mask[y0:y1, x0_adj:x1]
                chosen_mask = crop_mask
                chosen_class = "seed_fallback"

        if chosen_box is None or chosen_mask is None:
            continue

        plants.append(
            {
                "mask": chosen_mask,
                "bbox": (y0, y1, x0_adj, x1),
                "class": chosen_class,
            }
        )

    # save debug crops if requested
    if debug_save_dir is not None:
        Path(debug_save_dir).mkdir(parents=True, exist_ok=True)
        for idx, plant in enumerate(plants, 1):
            debug_path = Path(debug_save_dir) / f"{stem}_plant_{idx}.png"
            cv2.imwrite(str(debug_path), (plant["mask"] > 0).astype("uint8") * 255)

    return plants


def compute_root_lengths_for_image(
    full_mask,
    meta,
    debug_save_dir=None,
):
    """
    Split a full-image root mask into per-plant masks using per-column detector,
    then compute primary root length for each plant.
    Returns a list of rows with plant_id, length_px, tip_px, base_px, and RSA metadata.
    """
    if full_mask is None:
        return []

    stem = meta.get("stem", "image")

    rows = []

    plants = detect_plants_per_column(
        full_mask=full_mask,
        columns=5,
        debug_save_dir=debug_save_dir,
        stem=stem,
    )

    for idx, comp in enumerate(plants):
        mask_to_use = comp.get("mask")
        length_px, base_xy, tip_xy, graph_info = primary_length_px_from_mask(
            mask_to_use
        )
        plant_id = f"{stem}_plant_{idx+1}"
        rows.append(
            {
                "plant_id": plant_id,
                "length_px": length_px,
                "tip_px": tip_xy,
                "base_px": base_xy,
                "rsa": graph_info,
            }
        )
    return rows
