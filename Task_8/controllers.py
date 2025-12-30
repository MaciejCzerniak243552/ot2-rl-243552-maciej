from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class WorkspaceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    z_default: Optional[float] = None

    def __post_init__(self):
        if self.z_default is None:
            # default to the safer (higher) position within the envelope
            self.z_default = self.z_max


class PIDController:
    """
    Placeholder PID controller wrapper for pipeline integration.
    Replace internals with real robot API hooks when available.
    """

    def __init__(self, name: str = "PID"):
        self.name = name

    def move_to(self, target_xyz: Tuple[float, float, float]) -> Dict:
        return {"controller": self.name, "target": target_xyz, "status": "command_sent"}

    def inoculate(self) -> Dict:
        return {"controller": self.name, "action": "inoculate", "status": "command_sent"}


class RLController:
    """
    RL controller placeholder. Kept as TODO per instructions.
    """

    def __init__(self, name: str = "RL"):
        self.name = name

    def move_to(self, target_xyz: Tuple[float, float, float]) -> Dict:
        return {"controller": self.name, "target": target_xyz, "status": "todo"}

    def inoculate(self) -> Dict:
        return {"controller": self.name, "action": "inoculate", "status": "todo"}


def load_pid_controller() -> PIDController:
    return PIDController()


def load_rl_controller() -> RLController:
    return RLController()


def pixels_to_workspace(px: Tuple[int, int], calibration: Dict) -> Tuple[float, float, float]:
    """
    Simple affine pixel->workspace transform using scale + offset.
    """
    scale_x, scale_y = calibration.get("scale", (1.0, 1.0))
    off_x, off_y = calibration.get("offset", (0.0, 0.0))
    z_default = calibration.get("z_default", 0.0)
    x = px[0] * scale_x + off_x
    y = px[1] * scale_y + off_y
    return (x, y, z_default)


def validate_workspace(pt: Tuple[float, float, float], bounds: WorkspaceBounds) -> bool:
    x, y, z = pt
    return (
        bounds.x_min <= x <= bounds.x_max
        and bounds.y_min <= y <= bounds.y_max
        and bounds.z_min <= z <= bounds.z_max
    )
