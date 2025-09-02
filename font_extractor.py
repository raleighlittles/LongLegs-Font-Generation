import os
import logging
import numpy
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import argparse
import cv2
import sklearn
import scipy

logger = logging.getLogger(__name__)
logging.basicConfig(filename="font_extractor.log", encoding="utf-8", level=logging.DEBUG)

# ------------------------------
# Configurable parameters
# ------------------------------
THRESHOLD_SCALE = 0.6          # scale applied to mean for binarization threshold
MIN_COMPONENT_AREA = 150       # discard smaller connected components
COMPONENT_VERTICAL_TOLERANCE = 4
COMPONENT_HORIZONTAL_GAP = 12
GLYPH_PADDING = 8
GLYPH_BORDER = 2
HASH_SIZE = 18                 # size used for average hash
DEBUG_RECTANGLE_WIDTH = 3


def merge_band(band_components):
    """Merge touching/nearby components into glyph bounding boxes within a band."""
    band_components.sort(key=lambda x: (x[1], x[3]))
    merged_boxes = []

    for component in band_components:
        _, min_x, max_x, min_y, max_y = component
        placed = False
        for idx, (merge_x0, merge_x1, merge_y0, merge_y1) in enumerate(merged_boxes):
            vertical_overlap = not (max_y < merge_y0 - COMPONENT_VERTICAL_TOLERANCE or
                                    min_y > merge_y1 + COMPONENT_VERTICAL_TOLERANCE)
            horizontal_close = (min_x <= merge_x1 + COMPONENT_HORIZONTAL_GAP)
            if vertical_overlap and horizontal_close:
                merge_x0 = min(merge_x0, min_x)
                merge_x1 = max(merge_x1, max_x)
                merge_y0 = min(merge_y0, min_y)
                merge_y1 = max(merge_y1, max_y)
                merged_boxes[idx] = (merge_x0, merge_x1, merge_y0, merge_y1)
                placed = True
                break
        if not placed:
            merged_boxes.append((min_x, max_x, min_y, max_y))
    return merged_boxes


def average_hash(pil_image, hash_size=HASH_SIZE):
    """Simple perceptual average hash for deduplication."""
    resized_image = pil_image.resize((hash_size, hash_size), PIL.Image.LANCZOS).convert("L")
    pixel_array = numpy.array(resized_image, dtype=numpy.int16)
    mean_value = pixel_array.mean()
    bit_array = pixel_array > mean_value
    return bit_array.tobytes()

def extract_contour_points(img):
    """
    Takes a grayscale glyph image, returns contour points as Nx2 numpy array.
    """
    # Ensure binary
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return numpy.empty((0, 2))

    # Take the largest contour (assuming one glyph per image)
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]  # reshape to Nx2
    return contour.astype(float)

def extract_contour_from_image(img):
    """
    img: numpy array (grayscale or RGB). 
         White background, dark glyph works best.
    returns: Nx2 float32 array of contour points
    """
    if img.ndim == 3:  # convert RGB to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Threshold (invert so glyph=white)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return numpy.zeros((0, 2), dtype=numpy.float32)

    # Pick largest contour (by area)
    contour = max(contours, key=cv2.contourArea)

    # Flatten to Nx2
    contour = contour[:, 0, :].astype(numpy.float32)
    return contour

def center_scale_pca(points):
    P = points.astype(numpy.float32)
    if P.shape[0] == 0:
        return P
    # center
    c = P.mean(0, keepdims=True)
    P -= c
    # scale by bbox diagonal
    x0, y0 = P.min(0); x1, y1 = P.max(0)
    D = numpy.hypot(x1 - x0, y1 - y0) + 1e-8
    if D < 1e-3: # Guard against the case of very small bounding boxes
        return numpy.zeros((0, 2), dtype=numpy.float32)  # treat as empty
    P /= D
    # PCA orientation
    C = numpy.cov(P.T)
    eigvals, eigvecs = numpy.linalg.eigh(C)
    R = eigvecs[:, [1, 0]]  # reorder so major axis comes first
    P = P @ R
    # flip vertical if upside down
    if P[:, 1].mean() < 0:
        P[:, 1] *= -1
    return P

# --------------------------
# 3. Distances
# --------------------------
def modified_hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return numpy.inf
    D = sklearn.metrics.pairwise_distances(A, B)
    return max(D.min(axis=1).mean(), D.min(axis=0).mean())

def hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return numpy.inf
    D = sklearn.metrics.pairwise_distances(A, B)
    return max(D.min(axis=1).max(), D.min(axis=0).max())

# --------------------------
# 4. Utility to normalize from image
# --------------------------
def contour_from_image_normalized(img):
    contour = extract_contour_from_image(img)
    return center_scale_pca(contour)

def auto_threshold_nn(dist_matrix):
    """
    dist_matrix: symmetric NxN with zeros on the diagonal.
    Returns an unsupervised threshold via the knee in the 2-NN distances.
    """
    N = dist_matrix.shape[0]
    nn2 = []
    for i in range(N):
        row = numpy.sort(dist_matrix[i][dist_matrix[i] > 0])  # exclude self
        nn2.append(row[1] if len(row) > 1 else row[0])
    nn2_sorted = numpy.sort(nn2)
    # simple knee: largest discrete derivative
    diffs = numpy.diff(nn2_sorted)
    knee_idx = numpy.argmax(diffs)
    T = nn2_sorted[knee_idx]
    return float(T)

# ---------- Distance Measures ----------
def hausdorff_distance(contourA, contourB):
    """
    Symmetric Hausdorff distance between two contours.
    """
    d1 = scipy.spatial.distance.directed_hausdorff(contourA, contourB)[0]
    d2 = scipy.spatial.distance.directed_hausdorff(contourB, contourA)[0]
    return max(d1, d2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract glyphs from a font image.")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument("--num-rows", type=int, required=True,
                        help="Number of text rows in the screenshot.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"ERROR! Input image file not found: '{args.image}'")

    # Load image and convert to grayscale
    grayscale_image = PIL.Image.open(args.image).convert("L")
    grayscale_array = numpy.array(grayscale_image)

    # Threshold binarization
    threshold_value = int(numpy.mean(grayscale_array) * THRESHOLD_SCALE)
    binary_array = (grayscale_array > threshold_value).astype(numpy.uint8) * 255
    height, width = binary_array.shape
    logger.debug(f"Loaded image with size (height={height}, width={width}).")

    # Connected component labeling
    label_array = numpy.zeros((height, width), dtype=numpy.int32)
    current_label = 0
    for y_coord in range(height):
        for x_coord in range(width):
            if binary_array[y_coord, x_coord] and label_array[y_coord, x_coord] == 0:
                current_label += 1
                stack = [(y_coord, x_coord)]
                label_array[y_coord, x_coord] = current_label
                while stack:
                    cur_y, cur_x = stack.pop()
                    for neighbor_y, neighbor_x in (
                        (cur_y - 1, cur_x), (cur_y + 1, cur_x),
                        (cur_y, cur_x - 1), (cur_y, cur_x + 1)
                    ):
                        if (
                            0 <= neighbor_y < height and
                            0 <= neighbor_x < width and
                            binary_array[neighbor_y, neighbor_x] and
                            label_array[neighbor_y, neighbor_x] == 0
                        ):
                            label_array[neighbor_y, neighbor_x] = current_label
                            stack.append((neighbor_y, neighbor_x))

    # Bounding boxes
    component_boxes = {}
    for y_coord in range(height):
        for x_coord in range(width):
            label_id = label_array[y_coord, x_coord]
            if label_id:
                if label_id not in component_boxes:
                    component_boxes[label_id] = [x_coord, x_coord, y_coord, y_coord]
                box = component_boxes[label_id]
                box[0] = min(box[0], x_coord)
                box[1] = max(box[1], x_coord)
                box[2] = min(box[2], y_coord)
                box[3] = max(box[3], y_coord)

    logger.debug(f"Found {len(component_boxes)} connected components.")

    # Filter and sort components
    sorted_boxes = {
        k: v for k, v in sorted(
            component_boxes.items(),
            key=lambda item: (item[1][1] - item[1][0] + 1) * (item[1][3] - item[1][2] + 1),
            reverse=True
        )
    }

    component_list = []
    for label_id, (min_x, max_x, min_y, max_y) in sorted_boxes.items():
        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        if area >= MIN_COMPONENT_AREA:
            component_list.append((label_id, min_x, max_x, min_y, max_y))

    component_list.sort(key=lambda x: (x[3], x[1]))

    # Band segmentation by row
    y_centers = numpy.array([(min_y + max_y) / 2 for _, _, _, min_y, max_y in component_list])
    if len(y_centers) == 0:
        raise SystemExit("No components found; segmentation failed.")

    band_edges = numpy.linspace(y_centers.min(), y_centers.max(), args.num_rows)
    bands = [[] for _ in range(args.num_rows)]
    for component, y_center in zip(component_list, y_centers):
        band_index = numpy.searchsorted(band_edges[1:], y_center, side="right")
        band_index = min(band_index, args.num_rows - 1)
        bands[band_index].append(component)

    logger.debug(f"Segmented into {len(bands)} bands based on {args.num_rows} rows.")

    glyph_boxes = []
    for band in bands:
        glyph_boxes.extend(merge_band(band))

    # Extract glyphs
    glyph_images = []
    for min_x, max_x, min_y, max_y in glyph_boxes:
        min_x = max(min_x - GLYPH_PADDING, 0)
        min_y = max(min_y - GLYPH_PADDING, 0)
        max_x = min(max_x + GLYPH_PADDING, width - 1)
        max_y = min(max_y + GLYPH_PADDING, height - 1)

        cropped_array = binary_array[min_y:max_y + 1, min_x:max_x + 1]
        pil_glyph = PIL.Image.fromarray(cropped_array).convert("L")
        pil_glyph = PIL.ImageOps.invert(pil_glyph)
        pil_glyph = PIL.ImageOps.expand(pil_glyph, border=GLYPH_BORDER, fill=255)
        glyph_images.append(pil_glyph)

    logger.debug(f"Extracted {len(glyph_images)} glyph images.")

    # Save raw glyphs
    glyph_image_files = []
    for idx, glyph in enumerate(glyph_images):
        glyph_filename = f"glyph_img_{idx:02d}.png"
        glyph.save(glyph_filename)
        glyph_image_files.append(glyph_filename)

    # Debug visualization
    debug_image = grayscale_image.copy()
    drawer = PIL.ImageDraw.Draw(debug_image)
    for min_x, max_x, min_y, max_y in glyph_boxes:
        drawer.rectangle([min_x, min_y, max_x, max_y], outline="red", width=DEBUG_RECTANGLE_WIDTH)
    debug_image.save("bounding_boxes.png")

    # Deduplicate - step 1
    unique_glyphs = []
    seen_hashes = {}
    for glyph in glyph_images:
        glyph_hash = average_hash(glyph, hash_size=HASH_SIZE)
        if glyph_hash not in seen_hashes:
            seen_hashes[glyph_hash] = True
            unique_glyphs.append(glyph)

    contours = []
    for f in glyph_image_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        contour = contour_from_image_normalized(img)
        contours.append(contour)

    # Build distance matrix
    N = len(contours)
    D = numpy.zeros((N, N), dtype=numpy.float32)
    for i in range(N):
        for j in range(i+1, N):
            d = modified_hausdorff(contours[i], contours[j])
            D[i, j] = D[j, i] = d

    # Show matrix
    print("Pairwise Modified Hausdorff Distances:")
    print(D)

    import pdb; pdb.set_trace()

    # Simple threshold example
    THRESHOLD = 0.5
    for i in range(N):
        for j in range(i+1, N):
            if D[i, j] <= THRESHOLD:
                print(f"{glyph_image_files[i]} and {glyph_image_files[j]} are likely the same glyph.")
                # Put the two glyphs together into an image side-by-side for comparison
                glyphA = PIL.Image.open(glyph_image_files[i])
                glyphB = PIL.Image.open(glyph_image_files[j])
                combined = PIL.Image.new("L", (glyphA.width + glyphB.width, max(glyphA.height, glyphB.height)))
                combined.paste(glyphA, (0, 0))
                combined.paste(glyphB, (glyphA.width, 0))
                combined.save(f"comparison_{i}_{j}.png")

    unique_glyphs = [glyph for glyph in unique_glyphs if glyph is not None]

    # Save unique glyphs
    output_directory = os.path.join(os.getcwd(), "glyphs")
    os.makedirs(output_directory, exist_ok=True)
    glyph_paths = []
    for idx, glyph in enumerate(unique_glyphs):
        glyph_path = os.path.join(output_directory, f"glyph_{idx:02d}.png")
        glyph.save(glyph_path)
        glyph_paths.append(str(glyph_path))
