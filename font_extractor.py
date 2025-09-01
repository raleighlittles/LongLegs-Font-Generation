import os
import logging
import numpy
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import argparse

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
    for idx, glyph in enumerate(glyph_images):
        glyph.save(f"glyph_img_{idx:02d}.png")

    # Debug visualization
    debug_image = grayscale_image.copy()
    drawer = PIL.ImageDraw.Draw(debug_image)
    for min_x, max_x, min_y, max_y in glyph_boxes:
        drawer.rectangle([min_x, min_y, max_x, max_y], outline="red", width=DEBUG_RECTANGLE_WIDTH)
    debug_image.save("bounding_boxes.png")

    # Deduplicate
    unique_glyphs = []
    seen_hashes = {}
    for glyph in glyph_images:
        glyph_hash = average_hash(glyph, hash_size=HASH_SIZE)
        if glyph_hash not in seen_hashes:
            seen_hashes[glyph_hash] = True
            unique_glyphs.append(glyph)

    # Save unique glyphs
    output_directory = os.path.join(os.getcwd(), "glyphs")
    os.makedirs(output_directory, exist_ok=True)
    glyph_paths = []
    for idx, glyph in enumerate(unique_glyphs):
        glyph_path = os.path.join(output_directory, f"glyph_{idx:02d}.png")
        glyph.save(glyph_path)
        glyph_paths.append(str(glyph_path))
