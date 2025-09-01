# We'll attempt to:
# 1) Load the provided image with glyphs.
# 2) Segment individual glyphs by connected components.
# 3) Normalize and deduplicate glyphs by simple hash of resized bitmap.
# 4) Save each unique glyph as a monochrome PNG.
# 5) Build a bitmap OpenType font (CBDT/CBLC tables) mapped to PUA starting at U+E000.
# 6) Provide the .ttf for download.
#
# If fontTools is not available, we'll still save the unique glyph PNGs and a mapping CSV.

import os
import io
import math
import hashlib
import json
import zipfile
import sys
import traceback
import pathlib
import numpy
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import argparse


if __name__ == "__main__":

    # base_path = zipfile.Path("/mnt/data")
    # img_path = base_path / "vlcsnap-2025-06-16-22h49m47s707.png"

    parser = argparse.ArgumentParser(description="Extract glyphs from a font image.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--num-rows", help="Number of rows to segment the image into; based on the number of rows of text that are in the screenshot", type=int)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"ERROR! Input image file not found: '{args.image}'")

    # Load image
    img = PIL.Image.open(args.image).convert("L")
    # Invert (glyphs are light on dark)
    arr = numpy.array(img)
    # Normalize: threshold via Otsu-like heuristic
    thr = int(numpy.mean(arr) * 0.6)
    bw = (arr > thr).astype(numpy.uint8) * 255  # glyphs white
    # Morphological clean: remove thin noise with min area
    # We'll do connected-component labeling
    h, w = bw.shape

    labels = numpy.zeros((h, w), dtype=numpy.int32)
    label = 0
    # 4-connected components
    for y in range(h):
        for x in range(w):
            if bw[y, x] and labels[y, x] == 0:
                label += 1
                # BFS
                stack = [(y, x)]
                labels[y, x] = label
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and bw[ny,nx] and labels[ny,nx]==0:
                            labels[ny,nx] = label
                            stack.append((ny,nx))

    # Compute bounding boxes for each label
    boxes = {}
    for y in range(h):
        for x in range(w):
            l = labels[y,x]
            if l:
                if l not in boxes:
                    boxes[l] = [x,x,y,y] # minx,maxx,miny,maxy
                bx = boxes[l]
                if x < bx[0]: bx[0]=x
                if x > bx[1]: bx[1]=x
                if y < bx[2]: bx[2]=y
                if y > bx[3]: bx[3]=y

    # Sort boxes based on size (area) descending

    sorted_boxes  = {k: v for k, v in sorted(boxes.items(), key=lambda item: (item[1][1]-item[1][0]+1)*(item[1][3]-item[1][2]+1), reverse=True)}

    # Filter tiny components and merge those likely belonging to same glyph line-wise by proximity:
    # We'll start by simple: group components into glyphs by row bands using y-mid clustering.
    components = []
    for l, (minx,maxx,miny,maxy) in sorted_boxes.items():
        area = (maxx-minx+1)*(maxy-miny+1)
        if area < 150:  # skip tiny specks
            continue
        components.append((l, minx, maxx, miny, maxy))

    # Sort by top-left
    components.sort(key=lambda x: (x[3], x[1]))

    # Merge components that are close horizontally/vertically (parts of one glyph)
    # We'll sweep through left-to-right within each row band.
    glyph_boxes = []
    used = set()

    # Determine row bands using y centers
    ys = numpy.array([(miny+maxy)/2 for _,_,_,miny,maxy in components])
    if len(ys) == 0:
        raise SystemExit("No components found; segmentation failed.")

    # K simple banding - use the num_rows variable to determine band edges
    band_edges = numpy.linspace(ys.min(), ys.max(), args.num_rows)
    bands = [[] for _ in range(args.num_rows)]
    for comp, y in zip(components, ys):
        # find band index
        idx = numpy.searchsorted(band_edges[1:], y, side='right')
        if idx==args.num_rows: idx=args.num_rows-1
        bands[idx].append(comp)

    # Within each band, merge touching/nearby components into glyphs.
    # If one glyph is found inside of the bounds of another, do not treat them separately,
    # treat them as one glyph.
    def merge_band(band):
        band.sort(key=lambda x:(x[1],x[3]))
        merged = []
        for item in band:
            _, minx, maxx, miny, maxy = item
            placed = False
            for i, (mx0,mx1,my0,my1) in enumerate(merged):
                # if overlap vertically strongly and close horizontally, merge
                v_overlap = not (maxy < my0-4 or miny > my1+4)
                h_close = (minx <= mx1 + 12)  # gap threshold
                if v_overlap and h_close:
                    # merge
                    mx0 = min(mx0, minx); mx1 = max(mx1, maxx)
                    my0 = min(my0, miny); my1 = max(my1, maxy)
                    merged[i] = (mx0,mx1,my0,my1)
                    placed = True
                    break
            if not placed:
                merged.append((minx,maxx,miny,maxy))
        return merged

    for band in bands:
        glyph_boxes.extend(merge_band(band))

    # Extract glyph images with padding, normalize to common box
    pad = 8
    glyph_imgs = []
    for (minx,maxx,miny,maxy) in glyph_boxes:
        minx = max(minx - pad, 0); miny = max(miny - pad, 0)
        maxx = min(maxx + pad, w-1); maxy = min(maxy + pad, h-1)
        crop = bw[miny:maxy+1, minx:maxx+1]
        pil = PIL.Image.fromarray(crop).convert("L")
        # Trim and add margin to square
        pil = PIL.ImageOps.invert(pil)  # make glyph black on white
        # Resize into standard em square area e.g., 1000x1000 canvas with margins
        pil = PIL.ImageOps.expand(pil, border=2, fill=255)
        glyph_imgs.append(pil)

    # Save glyph images to file
    for i, im in enumerate(glyph_imgs):
        im.save(f"glyph_img_{i:02d}.png")

    # Create a version of the original image that has the bounding boxes drawn on it, for debugging
    debug_img = img.copy()
    draw = PIL.ImageDraw.Draw(debug_img)
    for (minx,maxx,miny,maxy) in glyph_boxes:
        draw.rectangle([minx, miny, maxx, maxy], outline="red", width=3)
    debug_img.save("bounding_boxes.png")

    # Deduplicate glyphs using perceptual hash (simple avg hash)
    def ahash(im, size=16):
        small = im.resize((size, size), PIL.Image.LANCZOS).convert("L")
        arr = numpy.array(small, dtype=numpy.int16)
        mean = arr.mean()
        bits = arr > mean
        return bits.tobytes()

    uniq = []
    seen = {}
    for im in glyph_imgs:
        key = ahash(im, size=18)
        if key in seen:
            continue
        seen[key] = True
        uniq.append(im)

    # Compare uniq and seen
    for key in seen.keys():
        if key not in [ahash(im, size=18) for im in uniq]:
            print(f"Duplicate glyph found: {key}")

    # Sort unique glyphs by approximate reading order (already roughly so)
    out_dir = os.getcwd() / "glyphs"
    out_dir.mkdir(exist_ok=True)
    glyph_paths = []
    for i, im in enumerate(uniq):
        p = out_dir / f"glyph_{i:02d}.png"
        im.save(p)
        glyph_paths.append(str(p))

    len(glyph_paths), glyph_paths[:5]
