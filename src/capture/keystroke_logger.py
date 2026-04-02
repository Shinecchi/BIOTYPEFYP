"""
keystroke_logger.py
--------------------
Captures live keystrokes from the physical keyboard using pynput.
Stores events as KeyEvent objects with high-precision timestamps.

Phase 1 — Behavioral Data Acquisition (FYP Proposal)
"""

import time
import threading
from typing import List, Optional
from pynput import keyboard

from src.utils.event_schema import KeyEvent


class KeystrokeLogger:
    """
    Listens to the physical keyboard and records all key press/release
    events with high-precision timestamps.

    Usage:
        logger = KeystrokeLogger()
        logger.start()
        # ... user types ...
        events = logger.stop_and_get_events()
    """

    def __init__(self):
        self._events: List[KeyEvent] = []
        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None
        self._running = False

    # ------------------------------------------------------------------
    # Keyboard Event Handlers
    # ------------------------------------------------------------------

    def _on_press(self, key):
        ts = time.perf_counter()
        key_str = self._normalize_key(key)
        with self._lock:
            self._events.append(KeyEvent(key=key_str, event_type='down', ts=ts))

    def _on_release(self, key):
        ts = time.perf_counter()
        key_str = self._normalize_key(key)
        with self._lock:
            self._events.append(KeyEvent(key=key_str, event_type='up', ts=ts))

    @staticmethod
    def _normalize_key(key) -> str:
        """Converts pynput Key objects to a consistent lowercase string."""
        try:
            # Regular character keys (a, b, c, etc.)
            return key.char.lower()
        except AttributeError:
            # Special keys (shift, ctrl, backspace, etc.)
            return str(key).lower()  # e.g. 'key.shift', 'key.backspace'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Starts the keyboard listener in a background thread."""
        if self._running:
            return
        self._events = []
        self._running = True
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop_and_get_events(self) -> List[KeyEvent]:
        """Stops listening and returns all captured events."""
        if self._listener:
            self._listener.stop()
        self._running = False
        with self._lock:
            return list(self._events)

    def get_events_snapshot(self) -> List[KeyEvent]:
        """Returns a snapshot of events so far WITHOUT stopping the listener."""
        with self._lock:
            return list(self._events)

    def clear(self):
        """Clears the event buffer (e.g. after a window has been processed)."""
        with self._lock:
            self._events = []

    @property
    def is_running(self) -> bool:
        return self._running
