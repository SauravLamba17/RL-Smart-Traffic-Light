"""
Vehicle model with realistic car-following kinematics and multi-tier emergency priority support.
Supports Standard Cars, Ambulances (Tier 2), and Fire Trucks (Tier 1).

Kinematics & collision-prevention guarantees:
- Realistic, smooth acceleration and progressive deceleration.
- Hard collision barrier preventing vehicle overlapping under all traffic loads.
- Strict red-light stop-line boundary preventing cars from creeping into red intersections.
"""
import random
from typing import Optional


# Vehicle Type Constants
VEHICLE_TYPE_CAR = 0
VEHICLE_TYPE_AMBULANCE = 1
VEHICLE_TYPE_FIRETRUCK = 2

# Emergency Priority Tiers:
# 0 -> Regular Civilian Vehicle
# 1 -> Tier 1 Emergency: Fire Truck (Highest Preemption Priority)
# 2 -> Tier 2 Emergency: Ambulance (Critical Medical Priority)
PRIORITY_TIER_NONE = 0
PRIORITY_TIER_FIRETRUCK = 1
PRIORITY_TIER_AMBULANCE = 2

# Pre-defined sleek vehicle color palette
CAR_COLORS = [
    (52, 152, 219),   # Blue
    (46, 204, 113),   # Green
    (241, 196, 15),   # Yellow
    (230, 126, 34),   # Orange
    (155, 89, 182),   # Purple
    (26, 188, 156),   # Turquoise
    (189, 195, 199),  # Silver
    (52, 73, 94),     # Dark Navy
]

AMBULANCE_COLOR = (255, 255, 255)       # White body
AMBULANCE_CROSS_COLOR = (231, 76, 60)   # Red cross
AMBULANCE_SIREN_1 = (255, 0, 0)         # Red flashing light
AMBULANCE_SIREN_2 = (0, 150, 255)       # Blue flashing light

FIRETRUCK_COLOR = (215, 25, 25)         # Signal Red body
FIRETRUCK_ACCENT = (245, 200, 30)       # Yellow reflective chevron
FIRETRUCK_ROOF = (180, 185, 195)        # Metallic roof ladder


class Vehicle:
    """
    Represents a single vehicle traveling through a 4-way intersection lane.

    Supports:
    - Standard Civilian Vehicles
    - Emergency Ambulances (Tier 2 Priority)
    - Emergency Fire Trucks (Tier 1 Priority - Heavy preemption)
    """

    def __init__(
        self,
        vehicle_id: int,
        lane: str,
        spawn_step: int,
        vehicle_type: int = VEHICLE_TYPE_CAR,
        position: float = 0.0,
        max_speed: float = 1.35,       # Realistic smooth cruise speed
        acceleration: float = 0.05,    # Smooth realistic acceleration
        deceleration: float = 0.09,    # Firm but smooth braking
    ):
        self.id = vehicle_id
        self.lane = lane  # 'N', 'S', 'E', 'W'
        self.spawn_step = spawn_step
        self.vehicle_type = vehicle_type
        self.position = position  # Rear bumper distance along lane from spawn (0.0)

        # Emergency properties & priority tier
        self.is_emergency = (vehicle_type in [VEHICLE_TYPE_AMBULANCE, VEHICLE_TYPE_FIRETRUCK])
        if vehicle_type == VEHICLE_TYPE_FIRETRUCK:
            self.priority_tier = PRIORITY_TIER_FIRETRUCK  # 1 (Highest)
            self.length = 34.0
            self.width = 16.0
            self.max_speed = max_speed * 1.15
            self.acceleration = acceleration * 1.20
            self.deceleration = deceleration * 1.25
            self.color = FIRETRUCK_COLOR
        elif vehicle_type == VEHICLE_TYPE_AMBULANCE:
            self.priority_tier = PRIORITY_TIER_AMBULANCE  # 2 (High)
            self.length = 26.0
            self.width = 15.0
            self.max_speed = max_speed * 1.25
            self.acceleration = acceleration * 1.30
            self.deceleration = deceleration * 1.30
            self.color = AMBULANCE_COLOR
        else:
            self.priority_tier = PRIORITY_TIER_NONE
            self.length = 22.0
            self.width = 14.0
            self.max_speed = max_speed + random.uniform(-0.08, 0.08)
            self.acceleration = acceleration
            self.deceleration = deceleration
            self.color = random.choice(CAR_COLORS)

        self.speed = 0.0               # Smooth rollout from spawn point
        self.wait_time = 0             # Timesteps spent stopped (speed < 0.1) before crossing
        self.total_time = 0            # Total timesteps in simulation
        self.has_cleared = False       # Set True when vehicle exits intersection
        self.cleared_step = -1
        self.is_braking = False
        self.committed = False         # True ONLY once vehicle has crossed stop line on green

    def update_physics(
        self,
        lead_vehicle: Optional["Vehicle"],
        is_green: bool,
        stop_line_pos: float,
        intersection_exit_pos: float,
        current_step: int,
    ) -> None:
        """
        Updates vehicle speed and position based on car-following kinematics and traffic light status.
        """
        if self.has_cleared:
            # Cruise smoothly off-screen
            self.position += self.max_speed
            return

        self.total_time += 1
        self.is_braking = False

        # Front bumper position
        front_pos = self.position + self.length

        # A vehicle is committed ONLY if it was green when crossing the stop line
        if is_green and front_pos > stop_line_pos and not self.committed:
            self.committed = True

        desired_speed = self.max_speed
        min_gap = 10.0  # Buffer gap behind lead vehicle

        # --- Rule 1: Car-following model ---
        if lead_vehicle is not None:
            lead_rear = lead_vehicle.position
            distance_to_lead = lead_rear - front_pos

            if distance_to_lead <= min_gap:
                desired_speed = 0.0
                self.is_braking = True
            elif distance_to_lead < min_gap * 3.5:
                ratio = (distance_to_lead - min_gap) / (min_gap * 2.5)
                ratio = max(0.0, min(1.0, ratio))
                leader_speed = max(lead_vehicle.speed, 0.0)
                desired_speed = min(desired_speed, leader_speed * ratio + (self.max_speed * ratio * 0.8))
                if desired_speed < self.speed:
                    self.is_braking = True

        # --- Rule 2: Red signal deceleration and stop ---
        if not is_green and not self.committed:
            distance_to_stop = stop_line_pos - front_pos
            if distance_to_stop <= 2.0:
                desired_speed = 0.0
                self.is_braking = True
            elif distance_to_stop < 65.0:
                # Progressive quadratic braking zone
                brake_ratio = max(0.0, distance_to_stop / 65.0)
                brake_speed = self.max_speed * (brake_ratio ** 1.3)
                desired_speed = min(desired_speed, brake_speed)
                if desired_speed < self.speed:
                    self.is_braking = True

        # --- Kinematic acceleration/deceleration ---
        if self.speed < desired_speed:
            self.speed = min(desired_speed, self.speed + self.acceleration)
        elif self.speed > desired_speed:
            self.speed = max(desired_speed, self.speed - self.deceleration)

        self.speed = max(0.0, self.speed)

        # Advance position
        self.position += self.speed

        # --- Strict physical anti-collision and red-light barriers ---
        # 1. Never overlap with lead vehicle
        if lead_vehicle is not None:
            max_safe_pos = lead_vehicle.position - self.length - 3.0
            if self.position > max_safe_pos:
                self.position = max_safe_pos
                self.speed = 0.0

        # 2. Never overshoot stop line on red
        if not is_green and not self.committed:
            max_stop_pos = stop_line_pos - self.length
            if self.position > max_stop_pos:
                self.position = max_stop_pos
                self.speed = 0.0

        # Track wait time if stopped
        if self.speed < 0.08 and not self.committed:
            self.wait_time += 1

        # Check if cleared intersection
        if self.position >= intersection_exit_pos and self.committed:
            self.has_cleared = True
            self.cleared_step = current_step

    def is_waiting(self, stop_line_pos: float) -> bool:
        """Returns True if vehicle is stopped/queued before the stop line."""
        return (not self.has_cleared) and (self.speed < 0.08) and (not self.committed)

    def is_in_queue(self, stop_line_pos: float) -> bool:
        """Returns True if vehicle is approaching or stopped at stop line."""
        return (not self.has_cleared) and (not self.committed)
