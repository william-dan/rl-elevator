from __future__ import annotations

"""Elevator Group‑Control Gymnasium Environment
------------------------------------------------
A discrete‑event elevator simulator based on:

    Wei et al., "Optimal Elevator Group Control via Deep Asynchronous
    Actor–Critic Learning", IEEE TNNLS 31 (12): 5245‑5256 (2020).

Key design choices
~~~~~~~~~~~~~~~~~~
* **Event‑driven time‑step** – Each *gym.step* advances to the next
  passenger/event as in the paper.  This makes the reward signal time‑
  homogeneous and avoids the need for arbitrary frame‑skips.
* **Tensor observation** – `obs[floor, car, channel]`, where the five
  stacked channels are   
  `B_↑`, `B_↓` (replicated hall‑calls, truncated by car direction),
  `A` (car‑calls), car **P**osition and **D**irection.
* **Capacity support** – Cars now enforce their `capacity`, boarding at
  most the available free slots.

This file is self‑contained – import it and register the environment via
`gymnasium.make("ElevatorEnv-v0")` if desired.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ────────────────────────────────────────────────────────────────────────────────
# Domain objects
# ────────────────────────────────────────────────────────────────────────────────


class Event(Enum):
    """Discrete events handled by the simulator."""

    CAR_DOOR_CLOSE = 0
    CAR_DOOR_OPEN = 1
    SPAWN_PASSENGER = 2


@dataclass
class Passenger:
    """State for a single passenger."""

    origin: int
    destination: int
    t_request: float
    t_board: Optional[float] = None
    t_alight: Optional[float] = None
    
    def __repr__(self) -> str:  # pragma: no cover – debug aid only
        return (
            "Passenger({o}→{d}, t={t:.2f})".format(
                o=self.origin,
                d=self.destination,
                t=self.t_request,
            )
        )


@dataclass
class Car:
    """Elevator car state."""

    capacity: int
    position: float = 0.0  # Continuous floor index (can be between floors)
    direction: int = 0  # −1, 0, +1
    door_open: bool = False
    t_door: float = 0.0  # Seconds until the current door cycle completes
    passengers: List[Passenger] = field(default_factory=list)
    itinerary: Optional[int] = None  # Next floor target (``None`` → idle)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    @property
    def remaining_capacity(self) -> int:
        """Return how many additional passengers *could* board."""

        if np.isinf(self.capacity):
            # Treat *inf* as an arbitrarily large integer to avoid spills in
            # list slicing logic downstream.
            return int(1e9)
        return max(self.capacity - len(self.passengers), 0)

    def __repr__(self) -> str:  # pragma: no cover – debug aid only
        return (
            "Car(pos={p:.2f}, dir={d:+d}, load={l}, it={it})".format(
                p=self.position,
                d=self.direction,
                l=len(self.passengers),
                it=self.itinerary,
            )
        )


# ────────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────────


class ElevatorEnv(gym.Env):
    """Multi‑car elevator control environment (discrete‑event)."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        *,
        num_floors: int = 10,
        num_cars: int = 4,
        capacity: int | float = np.inf,  # ∞ → unlimited
        speed_m_s: float = 1.5,
        floor_h_m: float = 3.5,
        door_time: float = 2.0,
        avg_passengers_spawning_time: float = 50.0,
        max_passengers_at_a_time: int = 10,
        avg_passengers_at_a_time: float = 5.0,
        total_passengers: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # ---------------- static parameters ------------------------------
        self.N: int = num_floors
        self.M: int = num_cars
        self.cap: float = capacity
        self.v: float = speed_m_s
        self.t_floor: float = floor_h_m / speed_m_s  # s / floor
        self.t_door_cycle: float = door_time  # s for a full open/close cycle

        self.avg_t_passenger: float = avg_passengers_spawning_time
        self.max_pax_batch: int = max_passengers_at_a_time
        self.avg_pax_batch: float = avg_passengers_at_a_time
        self.current_passengers: int = total_passengers
        self.total_passengers: int = total_passengers
        # ---------------- PRNG ------------------------------
        self.rng = np.random.default_rng(seed)

        # ---------------- Gym spaces ------------------------------
        # Observation: (floor, car, channel) – see module docstring
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.N, self.M, 5), dtype=np.float32
        )
        # Action: choose (floor, car) or `floor == N` → *do nothing*
        self.action_space = spaces.MultiDiscrete([self.N + 1, self.M])

        # ---------------- dynamic state ------------------------------
        self._reset_state()

    # ────────────────────────────────────────────────────────────────────
    # Gym API
    # ────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), self._info()

    def step(
        self, action: Sequence[int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        num_passengers_in_car = sum(
            len(c.passengers) for c in self.cars
        )
        num_passengers_in_hall = len(self.waiting)
        if num_passengers_in_car == 0 and num_passengers_in_hall == 0 and self.current_passengers == 0:
            return self._obs(), 1000.0, True, False, self._info()
        
        floor, car_idx = action
        car = self.cars[car_idx]

        if floor != self.N:  # *N* is the sentinel for "stay idle"
            car.itinerary = floor

        reward, event = self._advance_until_event()
        self.current_event = event
        # if len(self.done) > 0:
            # print(f"done:", self.done)
            
        return self._obs(), reward, False, False, self._info()

    # ────────────────────────────────────────────────────────────────────
    # Core simulation loop (discrete event)
    # ────────────────────────────────────────────────────────────────────

    def _advance_until_event(self) -> Tuple[float, Event]:
        """Simulate forward to the *next* event and return its reward + type."""

        self.new_request_this_step = None
        self.current_event = None

        # --- next Poisson arrival (absolute time) ----------------------
        if self.t_passenger_arrival is np.inf and self.current_passengers > 0:
            self.t_passenger_arrival = (
                self.rng.exponential(self.avg_t_passenger) + self.time
            )

        # --- next car event (relative time) ---------------------------
        t_car_next: float = np.inf  # relative seconds until any car event
        for car in self.cars:
            if car.door_open:
                t_car_next = min(t_car_next, car.t_door)
            elif car.itinerary is not None:
                t_car_next = min(
                    t_car_next,
                    abs(car.itinerary - car.position) * self.t_floor,
                )

        # --------------------------------------------------------------
        dt = min(self.t_passenger_arrival, t_car_next + self.time) - self.time
        if np.isinf(dt):
            # No events to process
            return self._reward_snapshot(), None
        self.time += dt

        event_fired: Optional[Event] = None

        # ---------------- update car states ---------------------------
        for car in self.cars:
            if car.door_open:
                # Countdown door timer
                car.t_door -= dt
                if car.t_door <= 1e-6:  # door now closes
                    car.door_open = False
                    car.t_door = 0.0
                    event_fired = Event.CAR_DOOR_CLOSE
            elif car.itinerary is not None:
                # Progress towards next stop
                sign = np.sign(car.itinerary - car.position)
                car.position += sign * (dt / self.t_floor)
                car.direction = int(sign)
                if abs(car.position - car.itinerary) < 1e-3:
                    # Arrived exactly – snap to floor
                    car.position = float(car.itinerary)
                    is_changed = self._handle_arrival(car, previous_direction=car.direction)
                    car.itinerary = None
                        
                    car.door_open, car.t_door = True, self.t_door_cycle
                    event_fired = Event.CAR_DOOR_OPEN
            else:
                car.direction = 0  # Idle

        # ---------------- passenger arrival (overrides) --------------
        if abs(self.t_passenger_arrival - self.time) < 1e-3:
            self._spawn_passenger()
            self.t_passenger_arrival = np.inf
            event_fired = Event.SPAWN_PASSENGER

        # assert event_fired is not None, "At least one event must fire each step"
        return self._reward_snapshot(), event_fired

    # ────────────────────────────────────────────────────────────────────
    # Passenger handling helpers
    # ────────────────────────────────────────────────────────────────────

    def _handle_arrival(self, car: Car, *, previous_direction: int) -> None:
        """Handle boarding/alighting when *car* reaches a floor."""

        floor = int(car.position)
        is_changed = False
        # ---------------- alight ------------------------------------------------
        for p in list(car.passengers):
            # print(p)
            if p.destination == floor:
                p.t_alight = self.time
                self.done.append(p)
                car.passengers.remove(p)
                self.arrival_reward += 2
                is_changed = True

        # ---------------- board -------------------------------------------------
        # Choose the direction the idle car *will* take – random tie‑break.
        if previous_direction == 0:
            previous_direction = int(self.rng.choice([-1, 1]))

        waiting_same_dir = [
            p
            for p in self.waiting
            if p.origin == floor and previous_direction * (p.destination - floor) > 0
        ]

        boardable = waiting_same_dir[: car.remaining_capacity]
        for p in boardable:
            p.t_board = self.time
            car.passengers.append(p)
            self.waiting.remove(p)
            self.board_reward += 1
            is_changed = True

        still_waiting = len(waiting_same_dir) > len(boardable)
        # if still_waiting:
            # print("Still waiting passengers:", waiting_same_dir)
        dir_col = 0 if previous_direction == 1 else 1
        self.hall[floor, dir_col] = int(still_waiting)
        
        return is_changed

    # ------------------------------------------------------------------

    def _spawn_passenger(self) -> None:
        """Generate a batch of new passengers (Poisson + exponential sizes)."""

        n_new = int(self.rng.exponential(self.avg_pax_batch) + 1)
        n_new = min(n_new, self.max_pax_batch)
        n_new = min(n_new, self.current_passengers)
        self.current_passengers -= n_new
        assert n_new > 0, "No new passengers to spawn"
        origin = self.rng.integers(0, self.N)
        for _ in range(n_new):
            destination = self.rng.choice([f for f in range(self.N) if f != origin])
            self.waiting.append(Passenger(origin, destination, self.time))
            self.new_request_this_step = origin
            self.hall[origin, 0 if destination > origin else 1] = 1

    # ────────────────────────────────────────────────────────────────────
    # Reward & observation utilities
    # ────────────────────────────────────────────────────────────────────

    def _reward_snapshot(self) -> float:
        """Negative waiting + riding time (scaled)."""
        # w = 0.0
        # for p in self.waiting:
        #     if p.t_request is not None:
        #         w += self.time - p.t_request
    
        # r = 0.0
        # for car in self.cars:
        #     for p in car.passengers:
        #         if p.t_board is not None:
        #             r += self.time - p.t_board
        # reward = -(w + r) * 1e-3 + self.arrival_reward + self.board_reward
        # self.arrival_reward = 0.0
        # self.board_reward = 0.0
        # print(f"reward: {-(w + r) * 1e-3:.2f}, w = {w:.2f}, r = {r:.2f}")
        reward = self.arrival_reward * 10 + self.board_reward
        self.arrival_reward = 0.0
        self.board_reward = 0.0
        return reward

    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        """Return the 3‑D tensor observation described in the docstring."""

        # Replicate hall calls per car and truncate against their direction
        B = np.repeat(self.hall.reshape(self.N, 2, 1), self.M, axis=2)
        for j, c in enumerate(self.cars):
            if c.direction > 0:
                B[: int(c.position), :, j] = 0  # below
            elif c.direction < 0:
                B[int(c.position) + 1 :, :, j] = 0  # above

        # Car‑calls (destinations)
        A = np.zeros((self.N, self.M), dtype=int)
        for j, c in enumerate(self.cars):
            for p in c.passengers:
                A[p.destination, j] = 1

        # Position & direction broadcast to N × M
        P = np.repeat(
            np.array([c.position for c in self.cars]).reshape(self.M, 1), self.N, 1
        ).T
        D = np.repeat(
            np.array([c.direction for c in self.cars]).reshape(self.M, 1), self.N, 1
        ).T

        stacked = np.stack([B[:, k, :] for k in range(2)] + [A, P, D], axis=2)
        return stacked.astype(np.float32)

    # ------------------------------------------------------------------

    def _info(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "waiting": len(self.waiting),
            "done": len(self.done),
            "hall_calls": self.hall.copy(),
            "cars": self.cars.copy(),
            "idle_cars": self._find_idle_cars(),
            "M": self.M,
            "N": self.N,
            "cars_itinerary": [c.itinerary for c in self.cars],
        }

    # ────────────────────────────────────────────────────────────────────
    # State helpers (public)
    # ────────────────────────────────────────────────────────────────────

    def _find_idle_cars(self) -> List[int]:
        return [i for i, c in enumerate(self.cars) if c.itinerary is None]

    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> None:  # noqa: D401 – Gym signature
        event_to_str = {
            Event.CAR_DOOR_CLOSE: "CLOSE",
            Event.CAR_DOOR_OPEN: "OPEN",
            Event.SPAWN_PASSENGER: "SPAWN",
        }
        color_map = {
            Event.CAR_DOOR_CLOSE: "\033[91m",  # red
            Event.CAR_DOOR_OPEN: "\033[92m",  # green
            Event.SPAWN_PASSENGER: "\033[93m",  # yellow
        }
        clr = color_map.get(self.current_event, "\033[0m")

        print(f"t={self.time:.1f}s | waiting={len(self.waiting)}")
        print(f"{clr}event: {event_to_str.get(self.current_event, '–')}\033[0m")
        for k, c in enumerate(self.cars):
            print(
                f" Car{k} floor={c.position:.2f} dir={c.direction:+d} load={len(c.passengers)} it={c.itinerary}"
            )
        print(" Hall‑calls:", np.where(self.hall.sum(1) > 0)[0].tolist())
        print("-" * 40)

    # ────────────────────────────────────────────────────────────────────
    # Internal reset helper
    # ────────────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.time: float = 0.0
        self.cars: List[Car] = [Car(self.cap) for _ in range(self.M)]
        self.hall: np.ndarray = np.zeros((self.N, 2), int)  # ↑ / ↓ hall calls
        self.waiting: List[Passenger] = []
        self.done: List[Passenger] = []
        self.current_event: Optional[Event] = None
        self.arrival_reward: float = 0.0
        self.board_reward: float = 0.0

        self.current_passengers: int = self.total_passengers
        self.t_passenger_arrival: Optional[float] = np.inf
        self.new_request_this_step: Optional[int] = None
