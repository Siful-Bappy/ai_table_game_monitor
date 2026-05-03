"""
DatasetAugmentor
================
Augments a per-class image dataset to fixed target counts for train/val splits.

Source layout expected:
    <source_dir>/
        <class_name>/
            img001.jpg
            img002.jpg
            ...

Output layout produced:
    <output_dir>/
        train/
            <class_name>/
                aug_0000.jpg
                ...
        val/
            <class_name>/
                val_0000.jpg
                ...

Extension points
----------------
Override any of these methods in a subclass to add new behaviour without
touching the core loop:

    get_transform_pipeline()  – add / remove augmentation steps
    preprocess_image()        – runs before augmentation  (e.g. resize to 224)
    postprocess_image()       – runs after augmentation   (e.g. normalise)

Usage
-----
    python augment_dataset.py
    python augment_dataset.py --source dataset_gkl_cards/organized \
                              --output dataset_gkl_cards/augmented \
                              --train-target 1200 --val-target 200
"""

import os
import re
import random
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np


class DatasetAugmentor:
    """
    Augments an image classification dataset to fixed per-class target counts.

    Parameters
    ----------
    source_dir   : root folder containing one sub-folder per class
    output_dir   : destination root; train/ and val/ sub-folders are created
    train_target : images per class in the training split  (default 1200)
    val_target   : images per class in the validation split (default 200)
    seed         : random seed for reproducibility
    """

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_target: int = 1200,
        val_target: int = 200,
        seed: int = 42,
        target_size: tuple = None,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_target = train_target
        self.val_target = val_target
        self.seed = seed
        # target_size = (width, height), e.g. (224, 224) for MobileNet
        # None means keep original image size
        self.target_size = target_size
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Extension hooks
    # ------------------------------------------------------------------

    def get_transform_pipeline(self) -> List[Callable]:
        """
        Returns the list of augmentation callables applied during training.

        Each callable receives an np.ndarray (BGR, H×W×C) and returns one.
        Each transform is applied independently with 50% probability.

        Override this method to add extra transforms (e.g. MixUp, CutOut)
        or to change which transforms are active.
        """
        return [
            self._random_rotation,
            # self._random_flip — DISABLED: flipping confuses 6↔9, J↔L, etc.
            self._random_brightness_contrast,
            self._random_hsv,
            self._random_gaussian_blur,
            self._random_gaussian_noise,
            self._random_perspective,
            self._random_crop_and_resize,
        ]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Runs on every image (train and val) BEFORE augmentation.

        If self.target_size is set, resizes the image to (width, height).
        Uses INTER_AREA for downscaling (better quality) and
        INTER_LINEAR for upscaling.
        Override this method in a subclass to add extra preprocessing.
        """
        if self.target_size is None:
            return image
        h, w = image.shape[:2]
        tw, th = self.target_size
        interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_LINEAR
        return cv2.resize(image, (tw, th), interpolation=interp)

    def postprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Runs on every image (train and val) AFTER augmentation.

        Currently a no-op — returns the image unchanged.
        Override for channel normalisation, format conversion, etc.
        """
        return image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self):
        """Process all class folders and write train/val splits.

        Uses a run-level 75/25 split: the last 25% of runs go entirely to val,
        the rest to train.  Runs are never split internally, so train and val
        see completely different card positions (no consecutive-frame leakage).
        """
        class_dirs = sorted(d for d in self.source_dir.iterdir() if d.is_dir())

        if not class_dirs:
            raise ValueError(f"No class sub-directories found in {self.source_dir}")

        print(f"Source  : {self.source_dir}")
        print(f"Output  : {self.output_dir}")
        print(f"Classes : {len(class_dirs)}")
        print(f"Target  : {self.train_target} train  /  {self.val_target} val  per class\n")

        for idx, class_dir in enumerate(class_dirs, 1):
            class_name = class_dir.name
            train_out = self.output_dir / 'train' / class_name
            val_out   = self.output_dir / 'val'   / class_name
            train_out.mkdir(parents=True, exist_ok=True)
            val_out.mkdir(parents=True, exist_ok=True)

            images = self._collect_images(class_dir)
            if not images:
                print(f"  [{idx:>3}/{len(class_dirs)}]  SKIP (no images): {class_name}")
                continue

            # Run-level split: assign entire runs to train or val.
            # Each run = consecutive frames of the same card in one position.
            # Splitting within a run leaks near-identical frames into val,
            # making accuracy trivially high. Keeping whole runs separate
            # ensures train and val see completely different card positions.
            runs = self._split_into_runs(images)
            n_val_runs  = max(1, round(len(runs) * 0.25))
            val_runs    = runs[-n_val_runs:]
            train_runs  = runs[:-n_val_runs] if len(runs) > n_val_runs else runs

            train_src = [p for run in train_runs for p in run]
            val_src   = [p for run in val_runs   for p in run]

            if not train_src:   # only 1 run total — use it for both
                train_src = val_src

            # Never cycle val images — every val image is unique
            val_actual = min(len(val_src), self.val_target)
            self._fill_split(val_src,   val_out,   val_actual,        augment=False, prefix='val')
            self._fill_split(train_src, train_out, self.train_target, augment=True,  prefix='aug')

            print(
                f"  [{idx:>3}/{len(class_dirs)}]  {class_name:<14} "
                f"src={len(images):>5}  runs={len(runs):>3}  "
                f"train_runs={len(train_runs):>2}  val_runs={len(val_runs):>2}  "
                f"val_imgs={val_actual}"
            )

        print("\nDone.")

    def check_image_sizes(self, samples_per_class: int = 3):
        """
        Reads a few images from each class and prints their sizes
        before and after preprocess_image() (i.e. after resize if target_size is set).
        Call this BEFORE generate() to verify your setup.
        """
        class_dirs = sorted(d for d in self.source_dir.iterdir() if d.is_dir())
        target_label = f"{self.target_size[0]}x{self.target_size[1]}" if self.target_size else "original"
        print(f"target_size = {self.target_size}  (resize to: {target_label})\n")
        print(f"{'CLASS':<16} {'FILE':<36} {'BEFORE':>14}  {'AFTER':>14}")
        print("-" * 84)
        for class_dir in class_dirs:
            images = self._collect_images(class_dir)
            if not images:
                continue
            for img_path in images[:samples_per_class]:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w, c = img.shape
                resized = self.preprocess_image(img)
                rh, rw, rc = resized.shape
                before = f"{h}x{w}x{c}"
                after  = f"{rh}x{rw}x{rc}"
                print(f"{class_dir.name:<16} {img_path.name:<36} {before:>14}  {after:>14}")
            print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_frame_number(path: Path) -> int:
        """Extract frame number from filenames like '018402_card03_c0.76.jpg'.
        Returns -1 if the name does not start with a numeric prefix."""
        m = re.match(r'^(\d+)_', path.name)
        return int(m.group(1)) if m else -1

    def _split_into_runs(self, images: List[Path]) -> List[List[Path]]:
        """
        Sort images by frame number and group them into consecutive-frame runs.
        A new run starts whenever the frame number is not exactly prev+1.

        Each run represents the card lying in one position on the table across
        multiple frames — all those frames look nearly identical.  Splitting by
        run (rather than random shuffle) ensures train and val see completely
        different card appearances.
        """
        tagged = sorted(
            [(self._get_frame_number(p), p) for p in images],
            key=lambda x: x[0],
        )

        runs: List[List[Path]] = []
        current: List[Path] = [tagged[0][1]]

        for i in range(1, len(tagged)):
            fn_prev, fn_curr = tagged[i - 1][0], tagged[i][0]
            # consecutive if frame numbers are adjacent AND both are valid
            if fn_prev != -1 and fn_curr != -1 and fn_curr == fn_prev + 1:
                current.append(tagged[i][1])
            else:
                runs.append(current)
                current = [tagged[i][1]]
        runs.append(current)
        return runs

    def _fill_split(
        self,
        src_images: List[Path],
        out_dir: Path,
        target: int,
        augment: bool,
        prefix: str,
    ):
        """Write exactly `target` images to `out_dir`, cycling src as needed."""
        n_src = len(src_images)
        for count in range(target):
            src_path = src_images[count % n_src]
            image = cv2.imread(str(src_path))
            if image is None:
                continue

            image = self.preprocess_image(image)
            if augment:
                image = self.augment_image(image)
            image = self.postprocess_image(image)

            out_path = out_dir / f"{prefix}_{count:04d}.jpg"
            cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply a random subset of the transform pipeline to one image."""
        for transform in self.get_transform_pipeline():
            if random.random() < 0.5:
                image = transform(image)
        return image

    def _collect_images(self, folder: Path) -> List[Path]:
        return [p for p in folder.iterdir() if p.suffix.lower() in self.IMAGE_EXTS]

    # ------------------------------------------------------------------
    # Individual transform implementations
    # ------------------------------------------------------------------

    def _random_rotation(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        angle = random.uniform(-10.0, 10.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _random_flip(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, random.choice([-1, 0, 1]))

    def _random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.6, 1.4)    # contrast multiplier
        beta  = random.randint(-40, 40)     # brightness offset
        out   = image.astype(np.float32) * alpha + beta
        return np.clip(out, 0, 255).astype(np.uint8)

    def _random_hsv(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-10,  10), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30,  30), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-30,  30), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _random_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def _random_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(5.0, 20.0)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _random_perspective(self, image: np.ndarray) -> np.ndarray:
        h, w   = image.shape[:2]
        margin = int(min(h, w) * 0.08)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.random.uniform(-margin, margin, src.shape).astype(np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _random_crop_and_resize(self, image: np.ndarray) -> np.ndarray:
        h, w      = image.shape[:2]
        frac      = random.uniform(0.80, 0.95)
        ch, cw    = int(h * frac), int(w * frac)
        y0        = random.randint(0, h - ch)
        x0        = random.randint(0, w - cw)
        cropped   = image[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# -----------------------------------------------------------------------
# Ready-to-use subclass — hard-wires 224 × 224 for MobileNet / EfficientNet
# -----------------------------------------------------------------------
class MobileNetAugmentor(DatasetAugmentor):
    """
    Convenience subclass that always resizes images to 224 × 224.

    Usage:
        aug = MobileNetAugmentor(
            source_dir='dataset_gkl_cards/organized',
            output_dir='dataset_gkl_cards/augmented_224',
        )
        aug.generate()
    """
    TARGET_SIZE = (224, 224)

    def __init__(self, source_dir: str, output_dir: str, **kwargs):
        # Force target_size to 224×224; caller cannot override it here
        kwargs['target_size'] = self.TARGET_SIZE
        super().__init__(source_dir, output_dir, **kwargs)


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Augment card image dataset to target counts.')
    parser.add_argument('--source',       default='dataset_gkl_cards/organized',
                        help='Root folder with one sub-folder per class')
    parser.add_argument('--output',       default='dataset_gkl_cards/augmented',
                        help='Destination root for train/ and val/ splits')
    parser.add_argument('--train-target', type=int, default=1200,
                        help='Images per class in training split')
    parser.add_argument('--val-target',   type=int, default=200,
                        help='Images per class in validation split')
    parser.add_argument('--seed',         type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--target-size',  type=int, nargs=2, default=[224, 224],
                        metavar=('W', 'H'),
                        help='Resize every image to W H (default: 224 224)')
    args = parser.parse_args()

    augmentor = DatasetAugmentor(
        source_dir=args.source,
        output_dir=args.output,
        train_target=args.train_target,
        val_target=args.val_target,
        seed=args.seed,
        target_size=tuple(args.target_size),
    )
    
    augmentor.generate()
    augmentor.check_image_sizes(samples_per_class=1)
