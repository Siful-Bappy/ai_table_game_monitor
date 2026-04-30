"""
VideoCompressionPipeline
========================
Re-encodes a test video at multiple H.264 CRF levels and reports file sizes.

This is used for the compression degradation study:
    Original video  →  re-encode at CRF [18, 23, 28, 35, 42, 51]
    Measure         :  file size, compression ratio, bitrate per CRF level

Output layout
-------------
    <output_dir>/
        <stem>_crf18.mp4
        <stem>_crf23.mp4
        <stem>_crf28.mp4
        <stem>_crf35.mp4
        <stem>_crf42.mp4
        <stem>_crf47.mp4
        <stem>_crf51.mp4
        size_vs_crf.png
        size_report.csv

Usage
-----
    # Default: re-encode gkl_mask1.avi at all CRF levels
    python compress_video.py

    # Custom input and output directory
    python compress_video.py --video gkl_mask1.avi --output compressed_videos

    # Custom CRF values
    python compress_video.py --video gkl_mask1.avi --crf 18 28 42

    # Preview one frame at each CRF level side-by-side
    python compress_video.py --video gkl_mask1.avi --preview
"""

import os
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np


class VideoCompressionPipeline:
    """
    Re-encodes a video at multiple H.264 CRF levels using ffmpeg.

    Parameters
    ----------
    video_path  : path to the source video file
    output_dir  : folder where re-encoded videos and reports are saved
    crf_levels  : list of H.264 CRF values  (0 = lossless, 51 = worst)
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = None,
        crf_levels: List[int] = None,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Default output dir: sibling folder named after the video stem
        if output_dir is None:
            output_dir = self.video_path.parent / f"{self.video_path.stem}_compressed"
        self.output_dir = Path(output_dir)
        self.crf_levels = crf_levels or [18, 23, 28, 35, 42, 51]
        self._check_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self):
        """Re-encode the video at every CRF level and print a size report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        original_size = self.video_path.stat().st_size
        info          = self._probe_video()

        print(f"Input      : {self.video_path}")
        print(f"Output dir : {self.output_dir}")
        print(f"Codec      : {info.get('codec', '?')}  "
              f"| Resolution : {info.get('width','?')}×{info.get('height','?')}  "
              f"| FPS : {info.get('fps','?')}  "
              f"| Frames : {info.get('frames','?')}")
        print(f"Original size : {_fmt_size(original_size)}\n")
        print(f"CRF levels : {self.crf_levels}\n")

        records = []

        for crf in self.crf_levels:
            out_path = self.output_dir / f"{self.video_path.stem}_crf{crf}.mp4"
            print(f"Encoding CRF {crf:>2}  →  {out_path.name} ...", end="", flush=True)

            ok = self._encode(self.video_path, out_path, crf)
            if not ok:
                print("  FAILED")
                continue

            enc_size  = out_path.stat().st_size
            ratio     = original_size / enc_size
            bitrate   = self._get_bitrate_kbps(out_path)
            records.append({
                "crf":           crf,
                "filename":      out_path.name,
                "size_mb":       round(enc_size / 1_048_576, 2),
                "ratio":         round(ratio, 2),
                "bitrate_kbps":  bitrate,
            })
            print(f"  {_fmt_size(enc_size):>10}  |  ratio {ratio:.1f}×  |  {bitrate} kbps")

        # ── Summary table ────────────────────────────────────────────
        print()
        print("=" * 65)
        print(f"{'CRF':<6} {'File':<30} {'Size':>8} {'Ratio':>8} {'Kbps':>8}")
        print("-" * 65)
        print(f"{'orig':<6} {self.video_path.name:<30} "
              f"{_fmt_size(original_size):>8}  {'1.0×':>7}  {'—':>7}")
        for r in records:
            print(f"{r['crf']:<6} {r['filename']:<30} "
                  f"{r['size_mb']:>6.1f}MB  {r['ratio']:>6.1f}×  {r['bitrate_kbps']:>6}")
        print("=" * 65)

        # ── Save CSV ─────────────────────────────────────────────────
        csv_path = self.output_dir / "size_report.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["crf", "filename", "size_mb", "ratio", "bitrate_kbps"])
            writer.writeheader()
            # Add original as first row
            writer.writerow({
                "crf": "original", "filename": self.video_path.name,
                "size_mb": round(original_size / 1_048_576, 2),
                "ratio": 1.0, "bitrate_kbps": self._get_bitrate_kbps(self.video_path),
            })
            writer.writerows(records)
        print(f"\nSaved → {csv_path}")

        # ── Plot ─────────────────────────────────────────────────────
        self._plot_size_curve(records, original_size)

        return records

    def preview(self, frame_index: int = 500):
        """
        Extract one frame from the original and each CRF-encoded video,
        then display them side-by-side.

        Press any key to close.
        """
        orig_frame = self._extract_frame(self.video_path, frame_index)
        if orig_frame is None:
            print(f"Could not extract frame {frame_index} from original.")
            return

        cv2.putText(orig_frame, "ORIGINAL", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        panels = [orig_frame]

        for crf in self.crf_levels:
            vid_path = self.output_dir / f"{self.video_path.stem}_crf{crf}.mp4"
            if not vid_path.exists():
                print(f"  SKIP crf_{crf}: {vid_path} not found — run generate() first.")
                continue
            frame = self._extract_frame(vid_path, frame_index)
            if frame is not None:
                label = (f"CRF {crf}  "
                         f"{_fmt_size(vid_path.stat().st_size)}")
                cv2.putText(frame, label, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                panels.append(frame)

        if not panels:
            print("No panels to display.")
            return

        # Resize all to same height before hconcat
        h = min(p.shape[0] for p in panels)
        w = min(p.shape[1] for p in panels)
        panels = [cv2.resize(p, (w, h)) for p in panels]

        # Display two rows if more than 4 panels
        if len(panels) <= 4:
            display = cv2.hconcat(panels)
        else:
            mid = len(panels) // 2
            row1 = cv2.hconcat(panels[:mid])
            row2 = cv2.hconcat(panels[mid:])
            # Pad shorter row to same width
            if row1.shape[1] != row2.shape[1]:
                max_w = max(row1.shape[1], row2.shape[1])
                row1 = _pad_width(row1, max_w)
                row2 = _pad_width(row2, max_w)
            display = cv2.vconcat([row1, row2])

        # Scale down if too wide for screen
        screen_w = 1920
        if display.shape[1] > screen_w:
            scale   = screen_w / display.shape[1]
            display = cv2.resize(display, (screen_w, int(display.shape[0] * scale)))

        cv2.imshow(f"Frame {frame_index} — Original vs CRF levels", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(src: Path, dst: Path, crf: int) -> bool:
        """Run ffmpeg H.264 encoding. Returns True on success."""
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(src),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-c:a", "copy",
            str(dst),
        ], capture_output=True)
        return result.returncode == 0

    @staticmethod
    def _extract_frame(video_path: Path, frame_index: int) -> np.ndarray:
        """Extract a single frame by index using cv2."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    @staticmethod
    def _get_bitrate_kbps(video_path: Path) -> int:
        """Return video bitrate in kbps via ffprobe."""
        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ], capture_output=True, text=True)
        try:
            bps = int(result.stdout.strip())
            return bps // 1000
        except (ValueError, AttributeError):
            return 0

    @staticmethod
    def _probe_video(video_path: Path = None) -> dict:
        """Return basic video info dict."""
        if video_path is None:
            return {}
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":    cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec":  "h264",
        }
        cap.release()
        return info

    def _probe_video(self) -> dict:
        cap = cv2.VideoCapture(str(self.video_path))
        info = {
            "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":    cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info

    def _plot_size_curve(self, records: list, original_size: int):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot.")
            return

        crfs  = [r["crf"]     for r in records]
        sizes = [r["size_mb"] for r in records]
        orig_mb = original_size / 1_048_576

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(crfs, sizes, marker='o', linewidth=2.5, color='steelblue',
                label="Re-encoded size (MB)")
        ax.axhline(orig_mb, color='tomato', linestyle='--', linewidth=1.5,
                   label=f"Original  ({orig_mb:.0f} MB)")

        # Annotate each point
        for crf, size in zip(crfs, sizes):
            ax.annotate(f"{size:.1f} MB",
                        xy=(crf, size), xytext=(0, 10),
                        textcoords="offset points", ha='center', fontsize=9)

        ax.set_xlabel("H.264 CRF  (←  high quality  |  low quality  →)", fontsize=12)
        ax.set_ylabel("File Size (MB)", fontsize=12)
        ax.set_title(f"File Size vs H.264 CRF Level\n{self.video_path.name}",
                     fontsize=13)
        ax.set_xticks(crfs)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plot_path = self.output_dir / "size_vs_crf.png"
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"Saved → {plot_path}")

    @staticmethod
    def _check_ffmpeg():
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(
                "ffmpeg not found. Install with:  sudo apt install ffmpeg"
            )


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def _fmt_size(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if n_bytes >= 1_073_741_824:
        return f"{n_bytes/1_073_741_824:.2f} GB"
    if n_bytes >= 1_048_576:
        return f"{n_bytes/1_048_576:.1f} MB"
    return f"{n_bytes/1024:.1f} KB"


def _pad_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Pad image on the right with black to reach target_w."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = np.zeros((h, target_w - w, 3), dtype=img.dtype)
    return np.hstack([img, pad])


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-encode a video at multiple H.264 CRF levels and report file sizes.")
    parser.add_argument("--video",   default="gkl_mask1.avi",
                        help="Input video file  (default: gkl_mask1.avi)")
    parser.add_argument("--output",  default=None,
                        help="Output directory  (default: <video_stem>_compressed/)")
    parser.add_argument("--crf", type=int, nargs="+",
                        default=[18, 23, 28, 35, 42, 47, 51],
                        help="CRF levels  (default: 18 23 28 35 42 51)")
    parser.add_argument("--preview", action="store_true",
                        help="Show frame comparison after generating")
    parser.add_argument("--frame",   type=int, default=500,
                        help="Frame index to use for preview  (default: 500)")
    args = parser.parse_args()

    pipeline = VideoCompressionPipeline(
        video_path=args.video,
        output_dir=args.output,
        crf_levels=args.crf,
    )
    pipeline.generate()

    if args.preview:
        pipeline.preview(frame_index=args.frame)
