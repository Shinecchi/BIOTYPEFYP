"""
trust_manager.py
-----------------
Phase 3 — Continuous Trust Evaluation & Decision Engine (FYP Proposal).

Converts per-window Euclidean distance scores from the Siamese Network
into a smooth, session-level trust score using Exponential Moving Average (EMA),
then makes access-control decisions via a 3-tier threshold system.

Trust Score:  1.0 = maximum confidence (same user)
              0.0 = minimum confidence (clear imposter)
Decision tiers:
  ACCESS GRANTED  : trust >= HIGH_THRESHOLD
  CHALLENGE       : LOW_THRESHOLD <= trust < HIGH_THRESHOLD
  ACCESS REVOKED  : trust < LOW_THRESHOLD
"""

from enum import Enum


class Decision(Enum):
    ACCESS_GRANTED  = "ACCESS_GRANTED"
    CHALLENGE       = "CHALLENGE"
    ACCESS_REVOKED  = "ACCESS_REVOKED"


class TrustManager:
    """
    Maintains a rolling EMA trust score from per-window distance values.
    Decision logic is applied on every update.

    Args:
        high_threshold  : Trust score above this = access granted. Default 0.60.
        low_threshold   : Trust score below this = access revoked. Default 0.35.
        ema_alpha       : EMA smoothing factor (0 < alpha <= 1). Default 0.3.
                         Lower = more stable (slower to react), Higher = more sensitive.
        max_distance    : Maximum expected Euclidean distance (used for normalisation).
    """

    def __init__(
        self,
        high_threshold: float = 0.60,
        low_threshold: float = 0.35,
        ema_alpha: float = 0.3,
        max_distance: float = 2.0,
    ):
        self.high_threshold = high_threshold
        self.low_threshold  = low_threshold
        self.ema_alpha      = ema_alpha
        self.max_distance   = max_distance

        # Session state
        self._trust_score: float = 1.0  # Start with full trust (user just enrolled)
        self._initialized: bool  = False
        self._window_count: int  = 0

    # ------------------------------------------------------------------
    # Core Update
    # ------------------------------------------------------------------

    def update(self, distance: float) -> Decision:
        """
        Feed a new per-window Euclidean distance from the Siamese Network.

        Args:
            distance : Raw distance between current embedding and enrolled profile.

        Returns:
            Decision enum — the current access-control decision.
        """
        # Convert distance → similarity (0 to 1)
        # Clamp distance to [0, max_distance] before normalising
        clamped = max(0.0, min(distance, self.max_distance))
        similarity = 1.0 - (clamped / self.max_distance)

        # Apply EMA smoothing
        if not self._initialized:
            self._trust_score = similarity
            self._initialized = True
        else:
            self._trust_score = (
                self.ema_alpha * similarity +
                (1.0 - self.ema_alpha) * self._trust_score
            )

        self._window_count += 1
        return self.current_decision

    # ------------------------------------------------------------------
    # Decision & State Access
    # ------------------------------------------------------------------

    @property
    def trust_score(self) -> float:
        """Current EMA trust score (0.0 – 1.0)."""
        return round(self._trust_score, 4)

    @property
    def current_decision(self) -> Decision:
        """Current 3-tier access-control decision based on trust score."""
        if self._trust_score >= self.high_threshold:
            return Decision.ACCESS_GRANTED
        elif self._trust_score >= self.low_threshold:
            return Decision.CHALLENGE
        else:
            return Decision.ACCESS_REVOKED

    def reset(self):
        """Reset trust state (e.g. on new session start)."""
        self._trust_score  = 1.0
        self._initialized  = False
        self._window_count = 0

    def summary(self) -> str:
        d = self.current_decision
        icon = {"ACCESS_GRANTED": "✅", "CHALLENGE": "⚠️", "ACCESS_REVOKED": "🚫"}.get(d.value, "?")
        return (
            f"{icon} [{d.value}]  Trust: {self.trust_score:.3f}  "
            f"(windows processed: {self._window_count})"
        )
