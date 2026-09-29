# TF-Flowers Fixture Attribution

The images in this directory are derived from:

- **Dataset:** TF-Flowers
- **Author:** Ye Xu
- **Version:** 1
- **DOI:** https://doi.org/10.6084/m9.figshare.19166516.v1
- **Source:** https://figshare.com/articles/dataset/TF-Flowers/19166516
- **License:** CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/)

HKDL selected the first 24 JPEG filenames in UTF-8 byte order from the
`daisy` and `sunflowers` classes. The first 16 files per class form the training
split and the next 8 form the evaluation split.

Each selected image was converted to RGB, center-cropped to a square, resized
to 64 by 64 pixels with bilinear interpolation, and encoded as JPEG at quality
90 with metadata removed. These transformations were made for the HKDL
offline training and evaluation fixture.

The exact source paths, split assignments, processed-file paths, and processed
SHA-256 digests are recorded in `manifest.json`.
