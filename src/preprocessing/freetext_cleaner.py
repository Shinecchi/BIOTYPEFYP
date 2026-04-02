"""
freetext_cleaner.py
-------------------
4-Stage cleaning engine for raw keystroke event streams.

Used for Generic (raw-event) datasets like Buffalo / IKDD.
(KeyRecs already provides pre-computed latencies, so it skips this module.)

Stages applied in order:
  Stage 1 — Modifier Key Filter
  Stage 2 — Backspace Burst Filter
  Stage 3 — Orphan Event Repair
  Stage 4 — Cognitive Pause Splitter  (splits into sub-sessions at >PAUSE_THRESHOLD_S gaps)

Input  : List[KeyEvent]  — one user's full raw event stream
Output : List[List[KeyEvent]]  — multiple clean sub-sessions per user
"""

from typing import List

from src.utils.event_schema import KeyEvent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keys that contribute NO biometric signal — remove all their events
MODIFIER_KEYS = {
    'shift', 'ctrl', 'alt', 'caps_lock', 'tab',
    'key.shift', 'key.shift_r', 'key.shift_l',
    'key.ctrl',  'key.ctrl_r',  'key.ctrl_l',
    'key.alt',   'key.alt_r',   'key.alt_l', 'key.alt_gr',
    'key.caps_lock', 'key.tab',
    'key.cmd',   'key.cmd_r',   'key.cmd_l',
    'key.super', 'key.meta',
    'key.windows',
}

# Keys that indicate error-correction (Backspace / Delete)
CORRECTION_KEYS = {'backspace', 'key.backspace', 'delete', 'key.delete'}

# Consecutive correction key threshold: runs >= this length are removed
CORRECTION_RUN_THRESHOLD = 3

# Maximum allowed gap between consecutive events before session is split (seconds)
PAUSE_THRESHOLD_S = 2.0


# ---------------------------------------------------------------------------
# Stage 1 — Modifier Key Filter
# ---------------------------------------------------------------------------

def _filter_modifiers(events: List[KeyEvent]) -> List[KeyEvent]:
    """Remove all events for modifier / coordination keys."""
    return [e for e in events if e.key.lower() not in MODIFIER_KEYS]


# ---------------------------------------------------------------------------
# Stage 2 — Backspace Burst Filter
# ---------------------------------------------------------------------------

def _filter_correction_bursts(events: List[KeyEvent]) -> List[KeyEvent]:
    """
    Remove runs of correction keys (Backspace/Delete) of length >= CORRECTION_RUN_THRESHOLD.
    A 'run' is a contiguous block of events whose key is a correction key.
    Single or double backspaces (typo fixes) are preserved — they are natural.
    """
    if not events:
        return events

    result: List[KeyEvent] = []
    i = 0
    while i < len(events):
        key_lower = events[i].key.lower()
        if key_lower in CORRECTION_KEYS:
            # Count the run
            run_start = i
            while i < len(events) and events[i].key.lower() in CORRECTION_KEYS:
                i += 1
            run_len = i - run_start
            if run_len < CORRECTION_RUN_THRESHOLD:
                # Short correction — keep it (natural typing behavior)
                result.extend(events[run_start:i])
            # else: long burst — discard entirely
        else:
            result.append(events[i])
            i += 1

    return result


# ---------------------------------------------------------------------------
# Stage 3 — Orphan Event Repair
# ---------------------------------------------------------------------------

def _repair_orphans(events: List[KeyEvent]) -> List[KeyEvent]:
    """
    Remove:
      - Any 'up' event with no preceding matching 'down'
      - Any 'down' event with no subsequent matching 'up'

    This handles window-boundary truncation from datasets.
    Uses a forward-pass + backward-pass approach.
    """
    # Forward pass: remove orphan 'up' events
    active_keys: set = set()
    forward: List[KeyEvent] = []
    for e in events:
        if e.event_type == 'down':
            active_keys.add(e.key)
            forward.append(e)
        elif e.event_type == 'up':
            if e.key in active_keys:
                active_keys.discard(e.key)
                forward.append(e)
            # else: orphan up — skip

    # Backward pass: remove 'down' events that never received an 'up'
    seen_up: set = set()
    backward: List[KeyEvent] = []
    for e in reversed(forward):
        if e.event_type == 'up':
            seen_up.add(e.key)
            backward.append(e)
        elif e.event_type == 'down':
            if e.key in seen_up:
                backward.append(e)
            # else: orphan down — skip

    backward.reverse()
    return backward


# ---------------------------------------------------------------------------
# Stage 4 — Cognitive Pause Splitter
# ---------------------------------------------------------------------------

def _split_at_pauses(events: List[KeyEvent]) -> List[List[KeyEvent]]:
    """
    Split the event stream into sub-sessions wherever the gap
    between consecutive event timestamps exceeds PAUSE_THRESHOLD_S.

    Returns a list of sub-sessions, each a List[KeyEvent].
    Empty sub-sessions (< 2 events) are discarded.
    """
    if not events:
        return []

    sub_sessions: List[List[KeyEvent]] = []
    current_session: List[KeyEvent] = [events[0]]

    for i in range(1, len(events)):
        gap = events[i].ts - events[i - 1].ts
        if gap > PAUSE_THRESHOLD_S:
            if len(current_session) >= 2:
                sub_sessions.append(current_session)
            current_session = [events[i]]
        else:
            current_session.append(events[i])

    if len(current_session) >= 2:
        sub_sessions.append(current_session)

    return sub_sessions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FreeTextCleaner:
    """
    Runs all 4 cleaning stages on a raw keystroke event stream.

    Usage:
        cleaner = FreeTextCleaner()
        sub_sessions = cleaner.clean(raw_events)
        # sub_sessions: List[List[KeyEvent]]
    """

    def __init__(
        self,
        pause_threshold_s: float = PAUSE_THRESHOLD_S,
        correction_run_threshold: int = CORRECTION_RUN_THRESHOLD,
    ):
        self.pause_threshold_s = pause_threshold_s
        self.correction_run_threshold = correction_run_threshold

    def clean(self, events: List[KeyEvent]) -> List[List[KeyEvent]]:
        """
        Apply all 4 stages in order and return clean sub-sessions.

        Args:
            events : Raw List[KeyEvent] for one user (sorted by ts).

        Returns:
            List[List[KeyEvent]] — multiple clean sub-sessions.
        """
        stats = {'original': len(events)}

        # Stage 1: Modifier filter
        events = _filter_modifiers(events)
        stats['after_modifier_filter'] = len(events)

        # Stage 2: Backspace burst filter
        events = _filter_correction_bursts(events)
        stats['after_burst_filter'] = len(events)

        # Stage 3: Orphan repair
        events = _repair_orphans(events)
        stats['after_orphan_repair'] = len(events)

        # Stage 4: Pause splitter
        sub_sessions = _split_at_pauses(events)
        stats['sub_sessions'] = len(sub_sessions)
        stats['total_clean_events'] = sum(len(s) for s in sub_sessions)

        return sub_sessions

    def clean_and_report(self, events: List[KeyEvent], user_id: str = '?') -> List[List[KeyEvent]]:
        """Same as clean() but prints a per-user summary."""
        original = len(events)
        sub_sessions = self.clean(events)
        total_clean = sum(len(s) for s in sub_sessions)
        dropped = original - total_clean
        print(
            f"  [Cleaner] User {user_id:>8}: "
            f"{original:>5} events → {total_clean:>5} clean "
            f"({dropped:>4} dropped, {len(sub_sessions)} sub-sessions)"
        )
        return sub_sessions
