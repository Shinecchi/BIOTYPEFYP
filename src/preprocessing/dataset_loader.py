"""
dataset_loader.py
-----------------
Loads one or more free-text keystroke datasets from CSV and returns
a unified structure:  Dict[user_id, List[KeyEvent]]

Supported formats out-of-the-box:
  - KeyRecs  : digraph latency CSV  (free-text.csv from Zenodo)
  - Generic  : raw event CSV        (for Buffalo / IKDD when available)

The KeyRecs format stores *pre-computed* inter-key timing columns:
  DD.k1.k2  =  Press(k2) - Press(k1)     [down-down]
  DU.k1.k2  =  Release(k2) - Press(k1)   [down-up]  --> covers Dwell of k1 implicitly
  UD.k1.k2  =  Press(k2) - Release(k1)   [up-down]  --> Flight Time  ✓
  UU.k1.k2  =  Release(k2) - Release(k1) [up-up]

We directly extract:
  Dwell  of key1  = DD.k1.k2  - ... approximated via  DU.k1.k2 [release k1 ref]
  A cleaner approach:
    Dwell  ≈  DD.k1.k2  (press-to-press minus flight ≈ hold duration)
    Flight =  UD.k1.k2  (release k1 → press k2)  ← the canonical flight time

So for each row we generate a (Dwell, Flight) feature pair directly,
then wrap them in synthetic KeyEvent-compatible containers for downstream processing.
"""

import os
import csv
import math
from typing import Dict, List, Optional, Tuple

from src.utils.event_schema import KeyEvent


# ---------------------------------------------------------------------------
# Internal lightweight container for pre-computed features
# Used only inside this loader; downstream modules consume List[KeyEvent]
# ---------------------------------------------------------------------------

class FeatureRow:
    """Holds a single extracted (dwell, flight) pair with user and session ID."""
    def __init__(self, user_id: str, session_id: str, dwell: float, flight: float):
        self.user_id = user_id
        self.session_id = session_id
        self.dwell = dwell    # seconds
        self.flight = flight  # seconds


# ---------------------------------------------------------------------------
# KeyRecs Loader  (digraph latency format)
# ---------------------------------------------------------------------------

# Thresholds for outlier removal (applied per feature)
_MIN_DWELL_S  = 0.020   # 20ms  — faster than human physiology
_MAX_DWELL_S  = 1.500   # 1.5s  — held key (modifier-like)
_MIN_FLIGHT_S = -0.200  # negative = key overlap (chorded), still valid
_MAX_FLIGHT_S = 2.000   # 2s pause threshold (AFK / cognitive break)


def load_keyrecs(
    csv_path: str,
    participant_col: str = 'participant',
    session_col: str = 'session',
    dwell_col_prefix: str = 'DD',    # we use DD as dwell proxy (see header comment)
    flight_col_prefix: str = 'UD',   # canonical flight time column prefix
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Reads a KeyRecs free-text.csv and extracts (dwell_s, flight_s) pairs per user.

    Returns:
        Dict[user_id_str : List[(dwell_s, flight_s)]]
        The list preserves row order (chronological within each session).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    user_features: Dict[str, List[Tuple[float, float]]] = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Find the first matching DD.* and UD.* column dynamically
        dd_cols = [h for h in headers if h.startswith(dwell_col_prefix + '.')]
        ud_cols = [h for h in headers if h.startswith(flight_col_prefix + '.')]

        if not dd_cols or not ud_cols:
            raise ValueError(
                f"Could not find '{dwell_col_prefix}.*' or '{flight_col_prefix}.*' "
                f"columns in {csv_path}. Available: {headers}"
            )

        for row in reader:
            uid = str(row.get(participant_col, '')).strip()
            if not uid:
                continue

            if uid not in user_features:
                user_features[uid] = []

            # Each row represents ONE digraph (key pair).
            # We iterate over every DD/UD column pair in the row.
            for dd_col, ud_col in zip(dd_cols, ud_cols):
                raw_dd = row.get(dd_col, '').strip()
                raw_ud = row.get(ud_col, '').strip()

                if not raw_dd or not raw_ud:
                    continue

                try:
                    dwell_s  = float(raw_dd)
                    flight_s = float(raw_ud)
                except ValueError:
                    continue

                # Skip NaN / Inf
                if not (math.isfinite(dwell_s) and math.isfinite(flight_s)):
                    continue

                # Apply outlier thresholds
                if not (_MIN_DWELL_S <= dwell_s <= _MAX_DWELL_S):
                    continue
                if not (_MIN_FLIGHT_S <= flight_s <= _MAX_FLIGHT_S):
                    continue

                user_features[uid].append((dwell_s, flight_s))

    print(f"[DatasetLoader] KeyRecs loaded: {len(user_features)} users from '{csv_path}'")
    for uid, feats in user_features.items():
        print(f"  User {uid:>6}: {len(feats):>5} feature pairs")

    return user_features


# ---------------------------------------------------------------------------
# Generic Raw-Event Loader  (for Buffalo / IKDD when available)
# ---------------------------------------------------------------------------

# Default column mapping for a generic raw-event dataset.
# Override these when calling load_raw_events() for your specific dataset.
DEFAULT_COLUMN_MAP = {
    'user_id':     'user_id',    # column name for participant/user identifier
    'key':         'key',        # column name for key character / keycode
    'event_type':  'event_type', # column name for press/release flag
    'timestamp':   'timestamp',  # column name for the raw timestamp
    # How event_type values are encoded in this dataset:
    'press_value': 'press',      # value meaning key-down  (e.g. 'press', 0, 'P')
    'release_value': 'release',  # value meaning key-up    (e.g. 'release', 1, 'R')
    # Timestamp unit: 'seconds' | 'milliseconds'
    'timestamp_unit': 'milliseconds',
}


def load_raw_events(
    csv_path: str,
    column_map: Optional[dict] = None,
) -> Dict[str, List[KeyEvent]]:
    """
    Reads a generic keystroke CSV with raw press/release events and
    returns Dict[user_id, List[KeyEvent]] sorted by timestamp per user.

    Args:
        csv_path   : Absolute path to the CSV file.
        column_map : dict overriding DEFAULT_COLUMN_MAP for your dataset.

    Returns:
        Dict[user_id_str : List[KeyEvent (sorted by ts)]]
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    cfg = {**DEFAULT_COLUMN_MAP, **(column_map or {})}

    user_events: Dict[str, List[KeyEvent]] = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            uid = str(row.get(cfg['user_id'], '')).strip()
            if not uid:
                continue

            key       = str(row.get(cfg['key'], '')).strip().lower()
            raw_type  = str(row.get(cfg['event_type'], '')).strip()
            raw_ts    = row.get(cfg['timestamp'], '').strip()

            if not key or not raw_type or not raw_ts:
                continue

            # Normalise event type
            press_val   = str(cfg['press_value']).strip()
            release_val = str(cfg['release_value']).strip()
            if raw_type == press_val or raw_type == '0':
                event_type = 'down'
            elif raw_type == release_val or raw_type == '1':
                event_type = 'up'
            else:
                continue  # unknown event type

            # Parse timestamp
            try:
                ts = float(raw_ts)
            except ValueError:
                continue

            # Convert to seconds
            if cfg['timestamp_unit'] == 'milliseconds':
                ts /= 1000.0

            uid_list = user_events.setdefault(uid, [])
            uid_list.append(KeyEvent(key=key, event_type=event_type, ts=ts))

    # Sort each user's events by timestamp
    for uid in user_events:
        user_events[uid].sort(key=lambda e: e.ts)

    print(f"[DatasetLoader] Raw events loaded: {len(user_events)} users from '{csv_path}'")
    return user_events
