"""
feature_extraction.py
---------------------
Converts a raw event window (List[KeyEvent]) into a feature matrix.

Output: List of [Dwell_Time_s, Flight_Time_s]

Follows FYP Proposal Phase 2 — 'timing characteristics derived from
raw key press and key release timestamps'.

Dwell Time  = time key is held down         (up.ts - down.ts)
Flight Time = time between consecutive keys  (down_next.ts - up_prev.ts)
              Negative values are valid (overlapping / chorded keystrokes).
"""

from typing import Dict, List, Optional

from src.utils.event_schema import KeyEvent


class FeatureExtractor:
    """
    Preprocessing Layer — Phase 2a.
    Converts a raw event window into [Dwell, Flight] feature pairs.
    """

    def extract_features(self, window: List[KeyEvent]) -> List[List[float]]:
        """
        Args:
            window : List[KeyEvent] — a single sliding-window segment.

        Returns:
            List of [dwell_s, flight_s] pairs.  Length <= window_size / 2.
        """
        features: List[List[float]] = []

        # Track when each key was pressed: {key_str: press_timestamp}
        key_down_ts: Dict[str, float] = {}

        # Release timestamp of the most recently completed keystroke
        last_key_up_ts: Optional[float] = None

        for event in window:
            if event.event_type == 'down':
                key_down_ts[event.key] = event.ts

            elif event.event_type == 'up':
                if event.key not in key_down_ts:
                    continue  # orphan up — skip

                down_ts = key_down_ts.pop(event.key)
                up_ts   = event.ts

                # 1. Dwell Time
                dwell = up_ts - down_ts

                # 2. Flight Time (Up-Down: release_prev → press_current)
                if last_key_up_ts is not None:
                    # We need the press time of THIS key to compute flight.
                    # down_ts already holds the press time for this key.
                    flight = down_ts - last_key_up_ts
                else:
                    # First completed keystroke in this window — no prior release
                    flight = 0.0

                last_key_up_ts = up_ts
                features.append([dwell, flight])

        return features
