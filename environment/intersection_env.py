"""
Intersection Simulation Environment for 4-way traffic control.
Supports multi-tier emergency vehicle preemption (Ambulances & Fire Trucks),
state discretization, and hierarchical priority reward shaping.
"""
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from environment.vehicle import (
    Vehicle,
    VEHICLE_TYPE_CAR,
    VEHICLE_TYPE_AMBULANCE,
    VEHICLE_TYPE_FIRETRUCK,
    PRIORITY_TIER_NONE,
    PRIORITY_TIER_FIRETRUCK,
    PRIORITY_TIER_AMBULANCE,
)


# Traffic Signal Phases
PHASE_NS_GREEN = 0   # North-South Green, East-West Red
PHASE_EW_GREEN = 1   # East-West Green, North-South Red
PHASE_NS_YELLOW = 2  # North-South Yellow transition
PHASE_EW_YELLOW = 3  # East-West Yellow transition
PHASE_ALL_RED = 4    # All directions red to clear intersection


class IntersectionEnv:
    """
    Simulated 4-Way Traffic Intersection Environment with Multi-Tier Emergency Support.
    """

    STOP_LINE_POS = 165.0          # Where vehicles stop on red
    INTERSECTION_EXIT_POS = 240.0  # Where vehicles clear the intersection
    ROAD_LENGTH = 440.0             # Total visual roadway span

    def __init__(
        self,
        max_steps: int = 1500,
        min_green_time: int = 150,      # 2.5 seconds (at 60 FPS)
        yellow_duration: int = 60,      # 1.0 second (at 60 FPS)
        arrival_rates: Optional[Dict[str, float]] = None,
        emergency_prob: float = 0.025,
        seed: Optional[int] = None,
    ):
        self.max_steps = max_steps
        self.min_green_time = min_green_time
        self.yellow_duration = yellow_duration
        self.emergency_prob = emergency_prob

        if arrival_rates is None:
            self.arrival_rates = {'N': 0.025, 'S': 0.025, 'E': 0.025, 'W': 0.025}
        else:
            self.arrival_rates = arrival_rates.copy()

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Simulation state
        self.current_step = 0
        self.current_phase = PHASE_NS_GREEN  # Agent observable phase (0: NS, 1: EW)
        self.signal_state = PHASE_NS_GREEN   # Internal signal state (includes yellow)
        self.phase_timer = 0                 # Timesteps in current phase
        self.yellow_timer = 0                # Timesteps remaining in yellow phase
        self.next_phase_after_yellow = PHASE_NS_GREEN
        self.in_clearance = False
        self.vehicle_id_counter = 0

        # Vehicle storage per lane
        self.lanes: Dict[str, List[Vehicle]] = {'N': [], 'S': [], 'E': [], 'W': []}

        # Metrics tracking
        self.total_spawned = 0
        self.total_cleared = 0
        self.total_ambulance_spawned = 0
        self.total_ambulance_cleared = 0
        self.total_firetruck_spawned = 0
        self.total_firetruck_cleared = 0
        self.cleared_wait_times: List[int] = []
        self.cleared_ambulance_wait_times: List[int] = []
        self.cleared_firetruck_wait_times: List[int] = []
        self.cumulative_reward = 0.0
        self.cumulative_wait_steps = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """Resets the environment to initial conditions."""
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.current_phase = PHASE_NS_GREEN
        self.signal_state = PHASE_NS_GREEN
        self.phase_timer = 0
        self.yellow_timer = 0
        self.next_phase_after_yellow = PHASE_NS_GREEN
        self.in_clearance = False
        self.vehicle_id_counter = 0

        self.lanes = {'N': [], 'S': [], 'E': [], 'W': []}

        self.total_spawned = 0
        self.total_cleared = 0
        self.total_ambulance_spawned = 0
        self.total_ambulance_cleared = 0
        self.total_firetruck_spawned = 0
        self.total_firetruck_cleared = 0
        self.cleared_wait_times.clear()
        self.cleared_ambulance_wait_times.clear()
        self.cleared_firetruck_wait_times.clear()
        self.cumulative_reward = 0.0
        self.cumulative_wait_steps = 0

        state = self.get_state()
        info = self._get_info()
        return state, info

    def set_traffic_scenario(self, scenario_name: str) -> None:
        """Sets pre-defined traffic arrival rate profiles."""
        if scenario_name == "balanced":
            self.arrival_rates = {'N': 0.025, 'S': 0.025, 'E': 0.025, 'W': 0.025}
        elif scenario_name == "rush_hour":
            self.arrival_rates = {'N': 0.045, 'S': 0.045, 'E': 0.045, 'W': 0.045}
        elif scenario_name == "asymmetric_ns":
            self.arrival_rates = {'N': 0.050, 'S': 0.050, 'E': 0.010, 'W': 0.010}
        elif scenario_name == "asymmetric_ew":
            self.arrival_rates = {'N': 0.010, 'S': 0.010, 'E': 0.050, 'W': 0.050}
        elif scenario_name == "light":
            self.arrival_rates = {'N': 0.012, 'S': 0.012, 'E': 0.012, 'W': 0.012}

    def spawn_emergency_vehicle(self, lane: str, vehicle_type: int = VEHICLE_TYPE_AMBULANCE) -> bool:
        """Manually spawns an emergency vehicle (Ambulance or Fire Truck) in the requested lane."""
        if lane not in self.lanes:
            return False

        # Ensure spawn point has adequate clearance
        if len(self.lanes[lane]) > 0 and self.lanes[lane][-1].position < 45.0:
            return False

        self.vehicle_id_counter += 1
        emerg_veh = Vehicle(
            vehicle_id=self.vehicle_id_counter,
            lane=lane,
            spawn_step=self.current_step,
            vehicle_type=vehicle_type,
            position=0.0
        )
        self.lanes[lane].append(emerg_veh)
        self.total_spawned += 1
        if vehicle_type == VEHICLE_TYPE_FIRETRUCK:
            self.total_firetruck_spawned += 1
        else:
            self.total_ambulance_spawned += 1
        return True

    def _spawn_vehicles(self) -> None:
        """Probabilistically spawns vehicles across all lanes based on arrival rates.
        
        IMPORTANT: All rng.random() calls are made unconditionally to keep
        the RNG sequence deterministic across environments with different
        traffic flow (e.g. Fixed-Timer vs RL), ensuring identical emergency
        vehicle spawns for fair side-by-side comparison.
        """
        for lane, rate in self.arrival_rates.items():
            if self.rng.random() < rate:
                # Always consume these random values to keep RNG in sync
                is_emergency = self.rng.random() < self.emergency_prob
                is_firetruck = self.rng.random() < 0.40

                # Check if spawn zone is free
                if len(self.lanes[lane]) == 0 or self.lanes[lane][-1].position > 45.0:
                    self.vehicle_id_counter += 1
                    
                    # Emergency vehicle generation (Ambulance or Fire Truck)
                    if is_emergency:
                        v_type = VEHICLE_TYPE_FIRETRUCK if is_firetruck else VEHICLE_TYPE_AMBULANCE
                        if v_type == VEHICLE_TYPE_FIRETRUCK:
                            self.total_firetruck_spawned += 1
                        else:
                            self.total_ambulance_spawned += 1
                    else:
                        v_type = VEHICLE_TYPE_CAR

                    vehicle = Vehicle(
                        vehicle_id=self.vehicle_id_counter,
                        lane=lane,
                        spawn_step=self.current_step,
                        vehicle_type=v_type,
                        position=0.0
                    )
                    self.lanes[lane].append(vehicle)
                    self.total_spawned += 1

    def discretize_count(self, count: int) -> int:
        """Discretizes queue count into 4 buckets: 0, 1-2, 3-5, 6+."""
        if count == 0:
            return 0
        elif count <= 2:
            return 1
        elif count <= 5:
            return 2
        else:
            return 3

    def discretize_timer(self, timer: int) -> int:
        """
        Discretizes phase timer into 3 buckets:
        0 -> recently switched / early green (< 150 steps = < 2.5s)
        1 -> steady green (150-359 steps = 2.5s - 6.0s)
        2 -> extended green (360+ steps = 6.0s+)
        """
        if timer < 150:
            return 0
        elif timer < 360:
            return 1
        else:
            return 2

    def get_queue_counts(self) -> Dict[str, int]:
        """Returns the number of queued vehicles before stop line in each lane."""
        counts = {}
        for lane, veh_list in self.lanes.items():
            waiting_cars = sum(1 for v in veh_list if v.is_in_queue(self.STOP_LINE_POS))
            counts[lane] = waiting_cars
        return counts

    def get_emergency_status(self) -> Tuple[int, int, int]:
        """
        Calculates emergency priority per direction:
        Returns:
            ns_prio: 0 = none, 1 = Ambulance (Tier 2), 2 = Fire Truck (Tier 1)
            ew_prio: 0 = none, 1 = Ambulance (Tier 2), 2 = Fire Truck (Tier 1)
            emerg_comb: combined index in [0, 8] (ns_prio * 3 + ew_prio)
        """
        ns_prio = 0
        for lane in ['N', 'S']:
            for v in self.lanes[lane]:
                if v.is_in_queue(self.STOP_LINE_POS):
                    if v.vehicle_type == VEHICLE_TYPE_FIRETRUCK:
                        ns_prio = max(ns_prio, 2)
                    elif v.vehicle_type == VEHICLE_TYPE_AMBULANCE:
                        ns_prio = max(ns_prio, 1)

        ew_prio = 0
        for lane in ['E', 'W']:
            for v in self.lanes[lane]:
                if v.is_in_queue(self.STOP_LINE_POS):
                    if v.vehicle_type == VEHICLE_TYPE_FIRETRUCK:
                        ew_prio = max(ew_prio, 2)
                    elif v.vehicle_type == VEHICLE_TYPE_AMBULANCE:
                        ew_prio = max(ew_prio, 1)

        emerg_comb = ns_prio * 3 + ew_prio
        return ns_prio, ew_prio, emerg_comb

    def get_state(self) -> Tuple[int, int, int, int, int, int, int]:
        """
        Returns 7-dimensional discretized state tuple:
        (cars_N, cars_S, cars_E, cars_W, current_phase, emerg_comb, timer_bucket)
        Total state space: 4 * 4 * 4 * 4 * 2 * 9 * 3 = 13,824 states
        """
        counts = self.get_queue_counts()
        b_n = self.discretize_count(counts['N'])
        b_s = self.discretize_count(counts['S'])
        b_e = self.discretize_count(counts['E'])
        b_w = self.discretize_count(counts['W'])
        _, _, emerg_comb = self.get_emergency_status()
        timer_bucket = self.discretize_timer(self.phase_timer)
        return (b_n, b_s, b_e, b_w, self.current_phase, emerg_comb, timer_bucket)

    @staticmethod
    def state_to_index(state: Tuple[int, ...]) -> int:
        """Maps 7-tuple state to unique integer in range [0, 13823]."""
        b_n, b_s, b_e, b_w, phase, emerg_comb, timer = state
        return (
            b_n * (4 * 4 * 4 * 2 * 9 * 3) +
            b_s * (4 * 4 * 2 * 9 * 3) +
            b_e * (4 * 2 * 9 * 3) +
            b_w * (2 * 9 * 3) +
            phase * (9 * 3) +
            emerg_comb * 3 +
            timer
        )

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, bool, Dict[str, Any]]:
        """
        Execute one simulation step.
        action: 0 = keep current phase, 1 = switch phase
        """
        self.current_step += 1
        self.phase_timer += 1
        phase_switched = False

        # Check if an emergency vehicle is waiting on RED (demands preemption switch)
        ns_prio, ew_prio, emerg_comb = self.get_emergency_status()
        emergency_on_red = False
        fire_on_red = False
        if self.current_phase == PHASE_NS_GREEN and ew_prio > 0:
            emergency_on_red = True
            fire_on_red = (ew_prio == 2)
        elif self.current_phase == PHASE_EW_GREEN and ns_prio > 0:
            emergency_on_red = True
            fire_on_red = (ns_prio == 2)

        # Preemption minimum green: 45 steps (0.75s) for Fire Truck, 60 steps (1.0s) for Ambulance vs min_green_time (150 steps = 2.5s)
        effective_min_green = 45 if fire_on_red else (60 if emergency_on_red else self.min_green_time)

        # Check intersection clearance
        intersection_clear = True
        for veh_list in self.lanes.values():
            for v in veh_list:
                if v.committed and v.position < self.INTERSECTION_EXIT_POS:
                    intersection_clear = False
                    break
            if not intersection_clear:
                break

        # Phase switching logic
        if self.yellow_timer > 0:
            # Transitioning through yellow light
            self.yellow_timer -= 1
            if self.yellow_timer == 0:
                self.in_clearance = True
                self.signal_state = PHASE_ALL_RED
        elif self.in_clearance:
            # Wait in ALL-RED state until intersection is fully clear
            if intersection_clear:
                self.in_clearance = False
                self.signal_state = self.next_phase_after_yellow
                self.current_phase = self.signal_state
                self.phase_timer = 0
        else:
            # Steady green phase
            if action == 1 and self.phase_timer >= effective_min_green:
                phase_switched = True
                if self.current_phase == PHASE_NS_GREEN:
                    self.signal_state = PHASE_NS_YELLOW
                    self.next_phase_after_yellow = PHASE_EW_GREEN
                else:
                    self.signal_state = PHASE_EW_YELLOW
                    self.next_phase_after_yellow = PHASE_NS_GREEN

                # Expedited yellow light during emergency preemption (30 steps = 0.5s for Fire, 45 steps = 0.75s for Amb)
                self.yellow_timer = 30 if fire_on_red else (45 if emergency_on_red else self.yellow_duration)

        # Spawn new vehicles
        self._spawn_vehicles()

        # Update vehicle physics across all lanes
        cleared_cars = 0
        cleared_amb = 0
        cleared_fire = 0

        is_ns_green = (self.signal_state == PHASE_NS_GREEN)
        is_ew_green = (self.signal_state == PHASE_EW_GREEN)

        for lane, veh_list in self.lanes.items():
            is_green = is_ns_green if lane in ['N', 'S'] else is_ew_green

            for i, veh in enumerate(veh_list):
                lead_veh = veh_list[i - 1] if i > 0 else None
                was_cleared = veh.has_cleared

                veh.update_physics(
                    lead_vehicle=lead_veh,
                    is_green=is_green,
                    stop_line_pos=self.STOP_LINE_POS,
                    intersection_exit_pos=self.INTERSECTION_EXIT_POS,
                    current_step=self.current_step
                )

                if not was_cleared and veh.has_cleared:
                    self.total_cleared += 1
                    self.cleared_wait_times.append(veh.wait_time)
                    if veh.vehicle_type == VEHICLE_TYPE_FIRETRUCK:
                        cleared_fire += 1
                        self.total_firetruck_cleared += 1
                        self.cleared_firetruck_wait_times.append(veh.wait_time)
                    elif veh.vehicle_type == VEHICLE_TYPE_AMBULANCE:
                        cleared_amb += 1
                        self.total_ambulance_cleared += 1
                        self.cleared_ambulance_wait_times.append(veh.wait_time)
                    else:
                        cleared_cars += 1

            # Purge vehicles that have exited the roadway
            self.lanes[lane] = [v for v in veh_list if v.position <= self.ROAD_LENGTH]

        # Count waiting vehicles per category
        waiting_cars = 0
        waiting_amb = 0
        waiting_fire = 0

        for veh_list in self.lanes.values():
            for v in veh_list:
                if v.is_waiting(self.STOP_LINE_POS):
                    if v.vehicle_type == VEHICLE_TYPE_FIRETRUCK:
                        waiting_fire += 1
                    elif v.vehicle_type == VEHICLE_TYPE_AMBULANCE:
                        waiting_amb += 1
                    else:
                        waiting_cars += 1

        total_waiting = waiting_cars + waiting_amb + waiting_fire
        self.cumulative_wait_steps += total_waiting

        # Queue imbalance penalty
        q_counts = self.get_queue_counts()
        ns_queue = q_counts['N'] + q_counts['S']
        ew_queue = q_counts['E'] + q_counts['W']
        imbalance_penalty = -0.05 * abs(ns_queue - ew_queue)

        # Multi-Tier Reward Shaping:
        # Fire Truck (Tier 1) > Ambulance (Tier 2) > Civilian Cars (Tier 3)
        emergency_cut_penalty = 0.0
        if phase_switched:
            if self.current_phase == PHASE_NS_GREEN and ns_prio > 0:
                emergency_cut_penalty = -60.0 if ns_prio == 2 else -35.0
            elif self.current_phase == PHASE_EW_GREEN and ew_prio > 0:
                emergency_cut_penalty = -60.0 if ew_prio == 2 else -35.0

        reward = (
            (10.0 * cleared_cars)
            + (100.0 * cleared_amb)
            + (200.0 * cleared_fire)
            - (0.05 * waiting_cars)
            - (1.5 * waiting_amb)
            - (3.0 * waiting_fire)
            - (1.0 if phase_switched else 0.0)
            + emergency_cut_penalty
            + imbalance_penalty
        )

        self.cumulative_reward += reward

        terminated = False
        truncated = (self.current_step >= self.max_steps)

        next_state = self.get_state()
        info = self._get_info()

        return next_state, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Collects summary simulation metrics converted to realistic real-world units (seconds, veh/min)."""
        queue_counts = self.get_queue_counts()
        # Convert wait time from frames to real seconds (60 frames/sec)
        avg_wait = (
            float(np.mean(self.cleared_wait_times)) / 60.0
            if len(self.cleared_wait_times) > 0
            else 0.0
        )
        avg_amb_wait = (
            float(np.mean(self.cleared_ambulance_wait_times)) / 60.0
            if len(self.cleared_ambulance_wait_times) > 0
            else 0.0
        )
        avg_fire_wait = (
            float(np.mean(self.cleared_firetruck_wait_times)) / 60.0
            if len(self.cleared_firetruck_wait_times) > 0
            else 0.0
        )
        # Convert throughput to vehicles per minute (3600 frames/min)
        throughput_per_min = (
            (self.total_cleared / max(1, self.current_step)) * 3600.0
        )
        ns_prio, ew_prio, emerg_comb = self.get_emergency_status()

        return {
            'step': self.current_step,
            'current_phase': self.current_phase,
            'signal_state': self.signal_state,
            'phase_timer': self.phase_timer,
            'queue_counts': queue_counts,
            'total_spawned': self.total_spawned,
            'total_cleared': self.total_cleared,
            'avg_wait_time': avg_wait,
            'avg_ambulance_wait_time': avg_amb_wait,
            'avg_firetruck_wait_time': avg_fire_wait,
            'throughput_per_min': float(throughput_per_min),
            'ambulance_spawned': self.total_ambulance_spawned,
            'ambulance_cleared': self.total_ambulance_cleared,
            'firetruck_spawned': self.total_firetruck_spawned,
            'firetruck_cleared': self.total_firetruck_cleared,
            'emergency_combined_status': emerg_comb,
            'cumulative_reward': self.cumulative_reward,
        }
