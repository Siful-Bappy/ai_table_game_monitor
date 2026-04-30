"""
CRFDegradationPipeline
======================
Generates H.264-degraded copies of the validation (and optionally training) set
at multiple CRF levels.

This is Step 1 of the compression degradation study:

    Train  : clean augmented card images  (dataset_gkl_cards/augmented/train)
    Test   : CRF-degraded val images      (dataset_gkl_cards/compressed_val/)
    Metric : Top-1 accuracy drop as CRF increases (image quality decreases)

Output layout
-------------
    dataset_gkl_cards/compressed_val/
        crf_18/  <class_name>/  img.jpg  ...
        crf_23/  ...
        crf_28/  ...
        crf_35/  ...
        crf_42/  ...
        crf_51/  ...

    dataset_gkl_cards/compressed_train/   (only with --train)
        crf_28/  <class_name>/  img.jpg  ...
        crf_38/  ...

H.264 CRF scale (libx264)
--------------------------
    0        lossless
    18       visually near-lossless  (high quality)
    23       default encoder quality
    28       moderate compression
    35       heavy compression
    42       very heavy compression
    51       worst quality

Usage
-----
    # Generate val CRF levels (default)
    python compress_val_dataset.py

    # Also generate compressed training set at CRF 28 and 38
    python compress_val_dataset.py --train

    # Custom val paths and CRF values
    python compress_val_dataset.py --source dataset_gkl_cards/augmented/val \\
                                   --output dataset_gkl_cards/compressed_val \\
                                   --crf 18 28 42

    # Custom train CRF values
    python compress_val_dataset.py --train --crf-train 28 38 42

    # Preview side-by-side before generating
    python compress_val_dataset.py --preview
"""

import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


class CRFDegradationPipeline:
    """
    Applies H.264 CRF compression to each image in a validation folder tree
    and saves degraded copies for every requested CRF level.

    Parameters
    ----------
    source_val_dir : folder with structure  <class_name>/<img>.jpg
    output_dir     : root where degraded datasets are written
    crf_levels     : list of H.264 CRF values  (0–51, lower = better quality)
    """

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(
        self,
        source_val_dir: str,
        output_dir: str,
        crf_levels: List[int] = None,
    ):
        self.source_dir = Path(source_val_dir)
        self.output_dir = Path(output_dir)
        self.crf_levels = crf_levels or [0, 18, 23, 28, 35, 42, 47, 51]
        self._check_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self):
        """Generate degraded val sets for every CRF level."""
        class_dirs = sorted(d for d in self.source_dir.iterdir() if d.is_dir())
        if not class_dirs:
            raise ValueError(f"No class sub-directories found in {self.source_dir}")

        print(f"Source     : {self.source_dir}")
        print(f"Output     : {self.output_dir}")
        print(f"Classes    : {len(class_dirs)}")
        print(f"CRF levels : {self.crf_levels}\n")

        for crf in self.crf_levels:
            print(f"── Generating CRF {crf} {'─'*44}")
            total_ok = 0

            for class_dir in tqdm(class_dirs, desc=f"CRF {crf:>2}"):
                out_class = self.output_dir / f"crf_{crf}" / class_dir.name
                out_class.mkdir(parents=True, exist_ok=True)

                images = [p for p in class_dir.iterdir()
                          if p.suffix.lower() in self.IMAGE_EXTS]

                for img_path in images:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    degraded = self.degrade_h264(img, crf)
                    if degraded is not None:
                        out_path = out_class / img_path.name
                        cv2.imwrite(str(out_path), degraded,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        total_ok += 1

            print(f"   {total_ok} images saved → {self.output_dir}/crf_{crf}/\n")

        print("Done.")

    def degrade_h264(self, img_bgr: np.ndarray, crf: int) -> np.ndarray:
        """
        Encode one image through H.264 libx264 at the given CRF,
        then decode the first frame back to BGR numpy array.

        Returns None if ffmpeg fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.png")
            vid_path = os.path.join(tmp, "compressed.mp4")
            out_path = os.path.join(tmp, "decoded.png")

            cv2.imwrite(src_path, img_bgr)

            # Encode single frame → H.264
            enc = subprocess.run([
                "ffmpeg", "-y",
                "-i", src_path,
                "-c:v", "libx264",
                "-crf", str(crf),
                "-frames:v", "1",
                vid_path,
            ], capture_output=True)

            if enc.returncode != 0:
                return None

            # Decode first frame → PNG
            dec = subprocess.run([
                "ffmpeg", "-y",
                "-i", vid_path,
                "-frames:v", "1",
                out_path,
            ], capture_output=True)

            if dec.returncode != 0:
                return None

            return cv2.imread(out_path)

    def preview(self, n_samples: int = 3):
        """
        Show side-by-side: original | CRF_18 | CRF_23 | ... for n_samples images.
        Press any key to advance, 'q' to quit.
        """
        all_imgs = [
            p for d in self.source_dir.iterdir() if d.is_dir()
            for p in d.iterdir() if p.suffix.lower() in self.IMAGE_EXTS
        ]
        samples = random.sample(all_imgs, min(n_samples, len(all_imgs)))

        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Label original
            orig = img.copy()
            cv2.putText(orig, "original", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            row = [orig]

            for crf in self.crf_levels:
                deg = self.degrade_h264(img, crf)
                if deg is not None:
                    cv2.putText(deg, f"CRF {crf}", (3, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    row.append(deg)

            strip = cv2.hconcat(row)
            cv2.imshow(f"Compression preview — {img_path.name}", strip)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ffmpeg():
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(
                "ffmpeg not found. Install with:  sudo apt install ffmpeg"
            )


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate H.264 CRF-degraded validation (and optionally training) sets.")
    # --- val args ---
    parser.add_argument("--source",  default="dataset_gkl_cards/augmented/val",
                        help="Clean val folder  (<class_name>/<img>.jpg)")
    parser.add_argument("--output",  default="dataset_gkl_cards/compressed_val",
                        help="Output root for degraded val datasets")
    parser.add_argument("--crf", type=int, nargs="+",
                        default=[0, 18, 23, 28, 35, 42, 47, 51],
                        help="CRF levels for val (default: 0 18 23 28 35 42 47 51)")
    # --- train args ---
    parser.add_argument("--train", action="store_true",
                        help="Also generate compressed training set")
    parser.add_argument("--source-train", default="dataset_gkl_cards/augmented/train",
                        help="Clean train folder  (default: dataset_gkl_cards/augmented/train)")
    parser.add_argument("--output-train", default="dataset_gkl_cards/compressed_train",
                        help="Output root for degraded train datasets")
    parser.add_argument("--crf-train", type=int, nargs="+", default=[28, 38],
                        help="CRF levels for train (default: 28 38)")
    # --- misc ---
    parser.add_argument("--preview", action="store_true",
                        help="Show side-by-side preview instead of generating files")
    args = parser.parse_args()

    # --- Val pipeline ---
    val_pipeline = CRFDegradationPipeline(
        source_val_dir=args.source,
        output_dir=args.output,
        crf_levels=args.crf,
    )

    if args.preview:
        val_pipeline.preview()
    else:
        val_pipeline.generate()

    # --- Train pipeline (optional) ---
    if args.train and not args.preview:
        print("\n" + "=" * 60)
        print("Training set compression")
        print("=" * 60)
        train_pipeline = CRFDegradationPipeline(
            source_val_dir=args.source_train,
            output_dir=args.output_train,
            crf_levels=args.crf_train,
        )
        train_pipeline.generate()
