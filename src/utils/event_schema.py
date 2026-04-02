from dataclasses import dataclass


@dataclass
class KeyEvent:
    """
    Raw keystroke event captured on the user device.

    key        : The key character. e.g. 'a', 'A', 'space', 'backspace'
    event_type : 'down' (key press) or 'up' (key release)
    ts         : Timestamp in seconds (monotonic, high-precision)
    """
    key: str
    event_type: str  # 'down' | 'up'
    ts: float        # seconds
