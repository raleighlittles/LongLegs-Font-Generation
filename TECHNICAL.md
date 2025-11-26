
## Problem statement and high level pipeline

Given a grayscale image $I(x,y)$ that contains several rows of handwritten glyphs, our goal is to produce a set of unique glyph images (one per distinct symbol) suitable for later font-building. The script follows:

1. Binarize the image to foreground / background.

2. **Connected components** to discover stroke blobs; compute bounding boxes.

3. **Row banding** (split components into the expected text rows) and merge nearby components into glyph boxes (to reattach broken strokes).

4. **Crop glyphs**, normalize and deduplicate:

* extract the outer contour of the glyph,

* apply a centering & scale normalization using PCA,

* compute pairwise shape distances (modified Hausdorff / symmetric Hausdorff) threshold or cluster to remove duplicates.

## Binarization (thresholding)

The first step converts an 8-bit grayscale image $I$ into a binary image $B$ where stroke pixels are 1 and background 0. The script uses a global threshold scaled from the image mean:

```math
T=α⋅mean(I) \\\\ B(x,y) = \begin{cases} 1 & I(x,y) > T \\ 0 & \texttt{otherwise.} \end{cases}
```

Here:

```math
\alpha = \textit{THRESHOLD\_SCALE}
```

There are two common alternatives worth knowing:

[Otsu’s method](https://en.wikipedia.org/wiki/Otsu%27s_method): finds $T$ that minimizes intra-class variance (used in some helper functions).

[Adaptive/local thresholding](https://en.wikipedia.org/wiki/Thresholding_(image_processing)): computes a local mean or median in a window and sets $T(x,y)=\text{localMean}(x,y)-C$. This is more robust to uneven illumination.

Note on preprocessing: a small median or Gaussian filter reduces salt-and-pepper noise and improves connected-component stability.

## Connected components and bounding boxes

After binarization we find connected foreground pixels using a 4-neighborhood BFS/DFS. Formally, we label each connected set $S_k \subset \mathbb{Z}^2$ and compute its bounding box:

```math
\text{bbox}_{k} = (x_{min}, x_{max}, y_{min}, y_{max}) \\
x_{min} = min_{(x,y) \in S_{k}} (x)
```

Small components are discarded by area ( MIN_COMPONENT_AREA ). The code's labeling is an $O(WH)$ scan for an $W\times H$ image.

## Row banding and merging sub-components into glyphs

Handwriting often produces glyphs that break into multiple nearby connected components (dots, diacritics, or stroke gaps). This script groups components into bands (rows) by comparing component vertical centers, then within each band merges components that:

have sufficient vertical overlap (tolerance COMPONENT_VERTICAL_TOLERANCE),

or are close horizontally (COMPONENT_HORIZONTAL_GAP).

This is implemented by scanning the band components left-to-right and greedily merging when the conditions hold. The merge operation simply computes the union of bounding boxes.

This heuristic is fast and effective for rows that are horizontally laid out and where glyphs are separated by gaps.

## Contour extraction and geometric normalization ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis))

For each glyph crop the code extracts a single (largest) contour using OpenCV (findContours) and then normalizes it with a PCA-based alignment.

After thresholding each glyph image into a binary mask, contours are sequences of points:

```math
C = \{(x_{1}, y_{1}), ... , (x_{m}, y_{m})\}
```

OpenCV’s findContours approximates stroke boundaries as point chains; picking the largest contour reduces spurious small components.

Let $P$ be the $m \times 2$ matrix of contour points. The code centers $P$ by subtracting the centroid $\bar p$:

```math
P_{c} = P - \bar{p}  \\ \\ \bar{p} = \frac{1}{m}\sum^{m}_{i=1} p_{i}
```

To normalize overall size, the code divides by the diagonal of the bounding box:

```math
D = \sqrt{(x_{max} - x_{min})^{2} + (y_{max} - y_{min})^2} \quad \quad P_{n} = \frac{P_{c}}{D}
```

This places contours in a canonical scale independent of writing size.

### PCA orientation

Compute the (2×2) sample [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix):

```math
C = \frac{1}{m} \sum^{m}_{i=1}(p_{i} - \bar{p})(p_{i} - \bar{p})^{T}
```

Solve the [eigen decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) $C v = \lambda v$. The eigenvectors $v_1, v_2$ define principal axes: rotating by the matrix $R=[v_1\ v_2]$ aligns the major axis horizontally. In the code this is used to rotate the points:

```math
P_{rot} = P_{n} R
```

Finally, a vertical flip is applied if the mean of the second coordinate is negative, ensuring a consistent “upright” orientation for similar glyphs.


## Automatic thresholding for clusters

The helper function `auto_threshold_nn` computes, for each glyph, the distance to its *second* nearest neighbor (2-NN). These 2-NN distances are then sorted from smallest to largest. When plotted, this sequence typically shows a long region of small values—corresponding to distances among visually similar or duplicate glyphs—followed by a sharp increase where distances begin reflecting dissimilar glyphs. The algorithm identifies the **largest jump** in this sorted list. The intuition is that items within the same group have consistently small mutual distances, while items from different groups are separated by noticeably larger distances. The largest gap therefore provides a data-driven threshold \(T\) that separates likely duplicates (distances < \(T\)) from distinct glyphs (distances ≥ \(T\)).

This is simple, fast and often effective when there are clear intra-/inter-class gaps.

## Perceptual hashing for quick deduplication

The script uses an average hash to remove obvious duplicates:

* Resize to a small grid (e.g. $H\times H$).

* Compute the mean $\mu$ of the pixel values.

* Create a bit mask $b_{ij} = 1$ if pixel $p_{ij}>\mu$ else $0$.

* Hash the bitmask bytes using SHA-1

## Limitations and Extensions

The pipeline relies on horizontal rows and readable spacing. For rotated or highly overlapping text one needs line-segmentation (Hough lines, projection profiles) or a trained object detector.