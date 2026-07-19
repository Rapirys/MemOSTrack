#!/usr/bin/env python3
"""Render selected GOT-10k test-set submission annotations as frame sequences."""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


DEFAULT_SEQUENCE_ROOT = Path("D:\\datasets\\Got10K\\test")
DEFAULT_SUBMIT_ROOT = Path("got10k_submit-60")
DEFAULT_OUTPUT_DIR = DEFAULT_SUBMIT_ROOT / "frames"
SEQUENCE_IDS = (12, 18, 29, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw GOT-10k submit-format boxes from *_001.txt files on test frames "
            "and save annotated frames for the hardcoded selected sequences."
        )
    )
    parser.add_argument(
        "--sequence_root",
        type=Path,
        default=DEFAULT_SEQUENCE_ROOT,
        help=f"Root containing GOT-10k test sequence folders. Default: {DEFAULT_SEQUENCE_ROOT}",
    )
    parser.add_argument(
        "--submit_root",
        type=Path,
        default=DEFAULT_SUBMIT_ROOT,
        help=f"Submission directory with per-sequence folders. Default: {DEFAULT_SUBMIT_ROOT}",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory for rendered frame folders. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness. Default: 2.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Output frame extension. Default: .jpg.",
    )
    return parser.parse_args()


def normalize_sequence_name(value: str) -> str:
    value = value.strip()
    if value.startswith("GOT-10k_Test_"):
        return value
    if value.isdigit():
        return f"GOT-10k_Test_{int(value):06d}"
    raise ValueError(
        f"Invalid sequence '{value}'. Use GOT-10k_Test_000001 or a numeric id."
    )


def load_bboxes(path: Path) -> np.ndarray:
    try:
        bboxes = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except ValueError:
        bboxes = np.loadtxt(path, dtype=np.float32)

    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)
    if bboxes.shape[1] != 4:
        raise ValueError(f"Expected Nx4 boxes, got shape {bboxes.shape} in {path}")
    return bboxes


def clipped_rectangle(
    bbox: np.ndarray, image_width: int, image_height: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x, y, w, h = bbox.tolist()
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    return (x1, y1), (x2, y2)


def render_sequence_frames(
    sequence_dir: Path,
    bbox_file: Path,
    output_dir: Path,
    thickness: int,
    image_ext: str,
) -> int:
    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No .jpg frames found in: {sequence_dir}")

    bboxes = load_bboxes(bbox_file)
    n_frames = min(len(frame_paths), len(bboxes))
    if len(frame_paths) != len(bboxes):
        print(
            f"[WARN] {sequence_dir.name}: {len(frame_paths)} frames but "
            f"{len(bboxes)} boxes. Rendering first {n_frames}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_path in enumerate(frame_paths[:n_frames]):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        height, width = frame.shape[:2]
        p1, p2 = clipped_rectangle(bboxes[frame_idx], width, height)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness)
        cv2.putText(
            frame,
            f"{sequence_dir.name}  {frame_idx + 1}/{n_frames}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = output_dir / f"{frame_path.stem}{image_ext}"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write frame: {output_path}")

    return n_frames


def main() -> None:
    args = parse_args()

    if args.thickness < 1:
        raise ValueError("--thickness must be >= 1")

    sequence_root = args.sequence_root.expanduser()
    submit_root = args.submit_root.expanduser()
    output_dir = args.output_dir.expanduser()
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"

    if not sequence_root.is_dir():
        raise FileNotFoundError(f"Sequence root not found: {sequence_root}")
    if not submit_root.is_dir():
        raise FileNotFoundError(f"Submission root not found: {submit_root}")

    sequence_names = [normalize_sequence_name(str(sequence_id)) for sequence_id in SEQUENCE_IDS]

    print(f"Rendering frame sequences: {', '.join(sequence_names)}")
    rendered = 0
    skipped = 0
    for sequence_name in sequence_names:
        sequence_dir = sequence_root / sequence_name
        bbox_file = submit_root / sequence_name / f"{sequence_name}_001.txt"
        sequence_output_dir = output_dir / sequence_name

        if not sequence_dir.is_dir():
            print(f"[WARN] Missing sequence dir: {sequence_dir}")
            skipped += 1
            continue
        if not bbox_file.is_file():
            print(f"[WARN] Missing bbox file: {bbox_file}")
            skipped += 1
            continue

        n_frames = render_sequence_frames(
            sequence_dir=sequence_dir,
            bbox_file=bbox_file,
            output_dir=sequence_output_dir,
            thickness=7,
            image_ext=image_ext,
        )
        rendered += 1
        print(f"[OK] {sequence_output_dir} ({n_frames} frames)")

    print(f"Done. Rendered {rendered}, skipped {skipped}.")


if __name__ == "__main__":
    main()
