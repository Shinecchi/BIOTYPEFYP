"""
build_training_data.py
----------------------
End-to-end orchestrator script.

PIPELINE:
  CSV (KeyRecs or raw-event)
    → DatasetLoader       (load & clean outliers)
    → [FreeTextCleaner]   (raw-event datasets only: modifier filter, burst filter,
                           orphan repair, pause splitter)
    → SlidingWindow       (20-event window, step 10 — 50% overlap)
    → FeatureExtractor    (dwell / flight per window)
    → LinearInterpolator  (→ fixed shape (10, 2))
    → GAFMATTransformer   (→ image (10, 10, 2))
    → Save .npy           (data/processed/{user_id}_images.npy)

USAGE:
  # KeyRecs dataset (digraph latency format):
  .\.venv\Scripts\python.exe -m src.scripts.build_training_data ^
      --dataset keyrecs ^
      --input   data/raw/free-text.csv ^
      --output  data/processed/

  # Generic raw-event dataset (Buffalo / IKDD):
  .\.venv\Scripts\python.exe -m src.scripts.build_training_data ^
      --dataset raw ^
      --input   data/raw/your_dataset.csv ^
      --output  data/processed/
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple

# Ensure project root is on the path when run as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.dataset_loader import load_keyrecs, load_raw_events
from src.preprocessing.freetext_cleaner import FreeTextCleaner
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.interpolation import LinearInterpolator
from src.preprocessing.gafmat import GAFMATTransformer
from src.utils.event_schema import KeyEvent


# ---------------------------------------------------------------------------
# Pipeline Hyperparameters (match FYP Proposal)
# ---------------------------------------------------------------------------
WINDOW_SIZE  = 20   # events per window  (10 keystrokes = 20 events)
STEP_SIZE    = 10   # overlap step       (50% overlap)
TARGET_LEN   = 10   # interpolation target length
IMAGE_SIZE   = 10   # GAFMAT output size → (10, 10, 2)
MIN_WINDOWS  = 5    # discard users with fewer than this many windows


# ---------------------------------------------------------------------------
# Sliding Window Helper
# ---------------------------------------------------------------------------

def sliding_window(events: List[KeyEvent], window_size: int, step: int):
    """Yield overlapping windows of `window_size` events."""
    start = 0
    while start + window_size <= len(events):
        yield events[start:start + window_size]
        start += step


# ---------------------------------------------------------------------------
# KeyRecs Pipeline (features already extracted as (dwell, flight) pairs)
# ---------------------------------------------------------------------------

def _run_keyrecs_pipeline(
    user_features: dict,
    interpolator: LinearInterpolator,
    gafmat: GAFMATTransformer,
    output_dir: str,
    min_windows: int,
) -> dict:
    """
    For KeyRecs: features are pre-computed (dwell, flight) pairs.
    We apply a sliding window over the pairs list, then interpolate + GAFMAT.
    """
    summary = {'users_processed': 0, 'users_skipped': 0, 'total_images': 0}

    for user_id, feat_pairs in user_features.items():
        images: List[np.ndarray] = []

        # Sliding window over feature pairs (each pair = 1 keystroke)
        start = 0
        while start + TARGET_LEN <= len(feat_pairs):
            window_pairs = feat_pairs[start:start + TARGET_LEN]
            start += (TARGET_LEN // 2)  # 50% overlap over feature pairs

            features_2d = [[d, f] for d, f in window_pairs]
            interpolated = interpolator.process(features_2d)  # (10, 2)
            gaf_image    = gafmat.transform(interpolated)      # (10, 10, 2)
            images.append(gaf_image)

        if len(images) < min_windows:
            print(f"  [Skip] User {user_id}: only {len(images)} windows (< {min_windows} min)")
            summary['users_skipped'] += 1
            continue

        _save_user(user_id, images, output_dir)
        summary['users_processed'] += 1
        summary['total_images'] += len(images)
        print(f"  [Saved] User {user_id}: {len(images)} images")

    return summary


# ---------------------------------------------------------------------------
# Raw-Event Pipeline (needs cleaning + segmentation + feature extraction)
# ---------------------------------------------------------------------------

def _run_raw_pipeline(
    user_events: dict,
    cleaner: FreeTextCleaner,
    extractor: FeatureExtractor,
    interpolator: LinearInterpolator,
    gafmat: GAFMATTransformer,
    output_dir: str,
    min_windows: int,
) -> dict:
    summary = {'users_processed': 0, 'users_skipped': 0, 'total_images': 0}

    for user_id, events in user_events.items():
        images: List[np.ndarray] = []

        # Clean the raw event stream
        sub_sessions = cleaner.clean_and_report(events, user_id=user_id)

        for session in sub_sessions:
            for window in sliding_window(session, WINDOW_SIZE, STEP_SIZE):
                raw_features = extractor.extract_features(window)
                if len(raw_features) < 2:
                    continue

                interpolated = interpolator.process(raw_features)  # (10, 2)
                gaf_image    = gafmat.transform(interpolated)       # (10, 10, 2)
                images.append(gaf_image)

        if len(images) < min_windows:
            print(f"  [Skip] User {user_id}: only {len(images)} windows (< {min_windows} min)")
            summary['users_skipped'] += 1
            continue

        _save_user(user_id, images, output_dir)
        summary['users_processed'] += 1
        summary['total_images'] += len(images)
        print(f"  [Saved] User {user_id}: {len(images)} images")

    return summary


# ---------------------------------------------------------------------------
# Save Utility
# ---------------------------------------------------------------------------

def _save_user(user_id: str, images: List[np.ndarray], output_dir: str):
    """Stack images and save as .npy file."""
    arr = np.stack(images, axis=0)   # (N, 10, 10, 2)
    safe_id = str(user_id).replace('/', '_').replace('\\', '_')
    out_path = os.path.join(output_dir, f"{safe_id}_images.npy")
    np.save(out_path, arr)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='BioType — Build Training Data from Free-Text Keystroke Dataset'
    )
    parser.add_argument(
        '--dataset', required=True, choices=['keyrecs', 'raw'],
        help='Dataset format: "keyrecs" (digraph latency CSV) or "raw" (press/release event CSV)'
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to the input CSV file (e.g. data/raw/free-text.csv)'
    )
    parser.add_argument(
        '--output', default='data/processed/',
        help='Output directory for .npy files (default: data/processed/)'
    )
    parser.add_argument(
        '--min-windows', type=int, default=MIN_WINDOWS,
        help=f'Minimum windows required per user (default: {MIN_WINDOWS})'
    )
    args = parser.parse_args()

    # Resolve paths relative to project root (cwd)
    input_path  = os.path.abspath(args.input)
    output_dir  = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  BioType — Training Data Builder")
    print("=" * 60)
    print(f"  Dataset format : {args.dataset}")
    print(f"  Input CSV      : {input_path}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Window size    : {WINDOW_SIZE} events  (step: {STEP_SIZE})")
    print(f"  Interpolation  : → ({TARGET_LEN}, 2)")
    print(f"  GAFMAT output  : → ({IMAGE_SIZE}, {IMAGE_SIZE}, 2)")
    print("=" * 60 + "\n")

    # Shared components
    interpolator = LinearInterpolator(target_length=TARGET_LEN)
    gafmat        = GAFMATTransformer(image_size=IMAGE_SIZE)

    if args.dataset == 'keyrecs':
        user_features = load_keyrecs(input_path)
        summary = _run_keyrecs_pipeline(
            user_features, interpolator, gafmat, output_dir, args.min_windows
        )

    else:  # raw
        user_events = load_raw_events(input_path)
        cleaner     = FreeTextCleaner()
        extractor   = FeatureExtractor()
        summary = _run_raw_pipeline(
            user_events, cleaner, extractor, interpolator, gafmat,
            output_dir, args.min_windows
        )

    # Final Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Users processed : {summary['users_processed']}")
    print(f"  Users skipped   : {summary['users_skipped']}")
    print(f"  Total images    : {summary['total_images']}")
    print(f"  Output dir      : {output_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
