"""
Fixed-Timer Traffic Light Controller (Baseline).
Switches signal phase strictly based on a pre-set clock duration.
"""
from typing import Tuple, Any


class FixedTimerController:
    """
    Baseline traffic controller that alternates phases at fixed intervals.
    """

    def __init__(self, phase_duration: int = 360):
        self.phase_duration = phase_duration
        self.name = "Fixed Timer (Baseline)"

    def get_action(self, env: Any, state: Tuple[int, ...] | None = None) -> int:
        """
        Determines action based strictly on phase timer.
        Returns:
            0: Keep current phase
            1: Switch phase
        """
        # If the environment has been in current phase for >= phase_duration, switch phase
        if env.phase_timer >= self.phase_duration:
            return 1
        return 0

    def reset(self) -> None:
        """Reset internal controller state if any."""
        pass
