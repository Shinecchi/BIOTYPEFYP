"""
main.py
-------
BioType Live Runtime — Phase 1 + 2 + 3 orchestrated together.

This is the entry point for live continuous authentication.

FLOW:
  1. User types for ~10 seconds during the ENROLLMENT phase.
     → Multiple GAFMAT images are generated from the typing session.
     → The Authenticator builds a 64D prototype embedding.

  2. Continuous VERIFICATION begins:
     → The keyboard logger captures live keystrokes.
     → A sliding window generates new GAFMAT images in real time.
     → Each image is compared to the prototype via Euclidean distance.
     → The TrustManager updates the EMA trust score.
     → A 3-tier decision is printed to the console.

Press ESC at any time to exit.
"""

import sys
import time
import threading
import os
import numpy as np
from pynput import keyboard as pynput_keyboard

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture.keystroke_logger       import KeystrokeLogger
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.interpolation      import LinearInterpolator
from src.preprocessing.gafmat             import GAFMATTransformer
from src.authenticator                     import BioTypeAuthenticator
from src.decision.trust_manager            import TrustManager, Decision

# ---------------------------------------------------------------------------
# Pipeline Config (matches training hyperparameters exactly)
# ---------------------------------------------------------------------------
WINDOW_SIZE      = 20   # events per window (10 keystrokes)
STEP_SIZE        = 10   # 50% overlap
ENROLL_IMAGES    = 10   # how many windows to collect during enrollment
ENROLL_PROMPT    = (
    "\n========================================\n"
    "  ENROLLMENT PHASE\n"
    "  Please type freely for ~10 seconds.\n"
    "  Press ENTER when done.\n"
    "========================================\n"
)

# Instantiate shared pipeline components
extractor    = FeatureExtractor()
interpolator = LinearInterpolator(target_length=10)
gafmat       = GAFMATTransformer(image_size=10)


# ---------------------------------------------------------------------------
# Helper: raw events → GAFMAT images using sliding window
# ---------------------------------------------------------------------------

def events_to_gafmat_images(events: list) -> list[np.ndarray]:
    images = []
    start = 0
    while start + WINDOW_SIZE <= len(events):
        window    = events[start:start + WINDOW_SIZE]
        raw_feats = extractor.extract_features(window)
        if len(raw_feats) < 2:
            start += STEP_SIZE
            continue
        interpolated = interpolator.process(raw_feats)
        img          = gafmat.transform(interpolated)
        images.append(img)
        start += STEP_SIZE
    return images


# ---------------------------------------------------------------------------
# Enrollment Phase
# ---------------------------------------------------------------------------

def run_enrollment(logger: KeystrokeLogger, auth: BioTypeAuthenticator) -> bool:
    print(ENROLL_PROMPT)
    logger.start()
    input()          # Wait for ENTER
    events = logger.stop_and_get_events()

    if len(events) < WINDOW_SIZE:
        print(f"[!] Not enough keystrokes captured ({len(events)} events). Please type more.")
        return False

    images = events_to_gafmat_images(events)
    if len(images) == 0:
        print("[!] Could not extract any GAFMAT windows from enrollment session.")
        return False

    print(f"[✓] Captured {len(images)} enrollment windows.")
    return auth.enroll(images)


# ---------------------------------------------------------------------------
# Verification Loop
# ---------------------------------------------------------------------------

_exit_flag = threading.Event()


def _on_escape(key):
    """ESC key → signal the verification loop to exit."""
    if key == pynput_keyboard.Key.esc:
        _exit_flag.set()
        return False  # Stop the pynput listener


def run_verification(auth: BioTypeAuthenticator, trust: TrustManager):
    print("\n========================================")
    print("  VERIFICATION PHASE — Monitoring...")
    print("  Press ESC to exit.")
    print("========================================\n")

    logger = KeystrokeLogger()
    logger.start()

    # ESC key listener (runs alongside)
    esc_listener = pynput_keyboard.Listener(on_press=_on_escape)
    esc_listener.start()

    # Sliding buffer — we process events as they accumulate
    processed_up_to = 0    # index into the logger's event buffer

    try:
        while not _exit_flag.is_set():
            time.sleep(0.5)     # check every 500ms

            events = logger.get_events_snapshot()
            total  = len(events)

            # Collect unprocessed events from the buffer
            new_events = events[processed_up_to:]

            while len(new_events) >= WINDOW_SIZE:
                window       = new_events[:WINDOW_SIZE]
                raw_feats    = extractor.extract_features(window)

                if len(raw_feats) >= 2:
                    interpolated = interpolator.process(raw_feats)
                    img          = gafmat.transform(interpolated)
                    distance     = auth.verify(img)
                    decision     = trust.update(distance)
                    _print_status(trust, decision, distance)

                # Slide forward
                new_events      = new_events[STEP_SIZE:]
                processed_up_to = total - len(new_events)

    except KeyboardInterrupt:
        pass
    finally:
        logger.stop_and_get_events()
        esc_listener.stop()
        print("\n[BioType] Session ended.")


def _print_status(trust: TrustManager, decision: Decision, distance: float):
    icon = {
        Decision.ACCESS_GRANTED: "✅",
        Decision.CHALLENGE:      "⚠️ ",
        Decision.ACCESS_REVOKED: "🚫",
    }[decision]
    print(
        f"{icon}  Trust: {trust.trust_score:.3f}   "
        f"Distance: {distance:.4f}   "
        f"Decision: {decision.value}"
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 40)
    print("   BioType — Keystroke Authentication")
    print("=" * 40)

    # Load the trained model
    try:
        auth = BioTypeAuthenticator()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    trust = TrustManager(
        high_threshold  = 0.75,   # Must stay above 75% similarity → stricter
        low_threshold   = 0.45,   # Below 45% → ACCESS_REVOKED
        ema_alpha       = 0.4,    # More reactive (default was 0.3)
        max_distance    = 0.80,   # Calibrated to actual distance range (0.01–0.67 observed)
    )

    # Enrollment
    logger = KeystrokeLogger()
    enrolled = False
    while not enrolled:
        enrolled = run_enrollment(logger, auth)
        if not enrolled:
            retry = input("[?] Retry enrollment? (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting.")
                sys.exit(0)

    # Verification loop
    run_verification(auth, trust)


if __name__ == '__main__':
    main()
