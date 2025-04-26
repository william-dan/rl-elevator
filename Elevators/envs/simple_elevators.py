"""
Elevator group‑control Gymnasium environment.
Implements the discrete‑event simulator from:

    Wei et al., "Optimal Elevator Group Control via Deep Asynchronous
    Actor–Critic Learning", IEEE TNNLS 31 (12): 5245‑5256 (2020)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ───────────────── data classes ──────────────────────────────────────────
@dataclass
class Passenger:
    origin: int
    destination: int
    t_request: float
    t_board: Optional[float] = None
    t_alight: Optional[float] = None

@dataclass
class Car:
    capacity: int
    position: float = 0.0          # continuous (floors)
    direction: int  = 0            # −1, 0, +1
    door_open: bool = False
    t_door: float   = 0.0          # time until door state change
    passengers: List[Passenger] = field(default_factory=list)
    # itinerary:  List[int] = field(default_factory=list)
    """
    itinerary can be changed at any time.
    """
    itinerary: int = None

# ───────────────── environment ──────────────────────────────────────────
class ElevatorEnv(gym.Env):
    """
    One RL *step* = next passenger event (arrival/board/alight) or
    a car finishing its door cycle — exactly the paper’s definition.
    """
    metadata = {"render_modes": ["human"],  "render_fps": 1}
    

    # ---------------- constructor ---------------------------------------
    def __init__(self,
                 num_floors: int = 10,
                 num_cars:   int = 4,
                 capacity:   int = np.inf,      # for simplicity
                 speed_m_s:  float = 1.5,
                 floor_h_m:  float = 3.5,
                 door_time:  float = 2.0,
                 passenger_rate: float = 0.2,    # λ (Poisson) per second
                 seed: int | None = None):

        self.N, self.M = num_floors, num_cars
        self.cap       = capacity
        self.v         = speed_m_s
        self.sec_floor = floor_h_m / speed_m_s
        self.t_door    = door_time
        self.lambda_p  = passenger_rate
        self.rng       = np.random.default_rng(seed)

        # observation = (N × M × 5) tensor flattened
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.N * self.M * 5,), dtype=np.float32)
        # action = one (floor,car) pair, add one for idle
        self.action_space = spaces.MultiDiscrete([self.N + 1, self.M])
        self._reset_state()

    # ---------------- Gymnasium API -------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), self._info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        floor, car_idx = action
        car = self.cars[car_idx]
        if floor != self.N:  # idle
            car.itinerary = floor
        # schedule if legal
        # if self._legal_stop(car, floor):
        #     car.itinerary.append(floor)

        reward = self._advance_until_event()      # discrete‑event simulation
        return self._obs(), reward, False, False, self._info()

    # ---------------- simulation core -----------------------------------
    def _advance_until_event(self) -> float:
        """Run until next passenger or door‑closure event, return reward."""
        # ––– time until next Poisson arrival –––
        t_pass = self.rng.exponential(1.0 / self.lambda_p)

        # ––– time until any car event –––
        t_car = np.inf
        for car in self.cars:
            if car.door_open:
                t_car = min(t_car, car.t_door)
            elif car.itinerary is not None:
                t_car = min(t_car,
                            abs(car.itinerary - car.position) * self.sec_floor)

        dt = min(t_pass, t_car)
        self.time += dt

        # move / count down doors
        for car in self.cars:
            if car.door_open:
                car.t_door -= dt
                if car.t_door <= 1e-6:
                    car.door_open, car.t_door = False, 0.0
                    return self._reward_snapshot()     # decision point
            elif car.itinerary is not None:
                sign = np.sign(car.itinerary - car.position)
                car.position += sign * (dt / self.sec_floor)
                car.direction = int(sign)
                if abs(car.position - car.itinerary) < 1e-3:
                    car.position = float(car.itinerary)
                    car.itinerary = None
                    self._handle_arrival(car)          # open doors
                    car.door_open, car.t_door = True, self.t_door
                    return self._reward_snapshot()     # decision point
            else:
                car.direction = 0

        # passenger arrival occurs first
        if t_pass < t_car:
            self._spawn_passenger()
            return self._reward_snapshot()             # decision point
        
        assert False, "Should never reach here"

    # ---------------- boarding/alighting & generator ---------------------
    def _handle_arrival(self, car: Car):
        floor = int(car.position)
        # alight
        for p in list(car.passengers):
            if p.destination == floor:
                p.t_alight = self.time
                self.done.append(p)
                car.passengers.remove(p)
        # board
        waiting_here = [p for p in self.waiting if p.origin == floor]
        space = self.cap - len(car.passengers)
        for p in waiting_here[:space]:
            p.t_board = self.time
            car.passengers.append(p)
            self.waiting.remove(p)
            # if p.destination not in car.itinerary:
            #     car.itinerary.append(p.destination)
        self.hall[floor, :] = 0  #! if capacity is limited, should not be like this

    def _spawn_passenger(self):
        o = self.rng.integers(0, self.N)
        d = self.rng.choice([f for f in range(self.N) if f != o])
        self.waiting.append(Passenger(o, d, self.time))
        self.hall[o, 0 if d > o else 1] += 1

    # ---------------- reward --------------------------------------------
    def _reward_snapshot(self):
        w = sum((self.time - p.t_request)**2 for p in self.waiting)
        r = sum((self.time - p.t_board)**2
                for car in self.cars for p in car.passengers if p.t_board)
        return -(w + r)

    # ---------------- utils ---------------------------------------------
    def _legal_stop(self, car: Car, floor: int) -> bool:
        if car.direction == 0:                     # idle
            return True
        return (car.direction > 0 and floor >= car.position) or \
               (car.direction < 0 and floor <= car.position)

    def _obs(self):
        # replicate & truncate hall‑calls -> B̄
        B = np.repeat(self.hall.reshape(self.N, 2, 1), self.M, axis=2)
        for j, c in enumerate(self.cars):
            if c.direction > 0:
                B[: int(c.position), :, j] = 0
            elif c.direction < 0:
                B[int(c.position)+1 :, :, j] = 0
        # car‑call matrix A
        A = np.zeros((self.N, self.M), dtype=int)
        for j, c in enumerate(self.cars):
            for p in c.passengers:
                A[p.destination, j] = 1
        # positions & directions stretched to N×M
        P = np.repeat(np.array([c.position for c in self.cars]).reshape(len(self.cars), 1), self.N, 1).T
        D = np.repeat(np.array([c.direction for c in self.cars]).reshape(len(self.cars), 1), self.N, 1).T
        # print(f"P shape: {P.shape}, D shape: {D.shape}")
        # print(f"B shape: {B.shape}, A shape: {A.shape}")
        stacked = np.stack([B[:, j, :] for j in range(2)] + [A, P, D], 2)
        # print(f"stacked shape: {stacked.shape}")
        return stacked.astype(np.float32).flatten()

    def _reset_state(self):
        self.time = 0.0
        self.cars    = [Car(self.cap) for _ in range(self.M)]
        self.hall    = np.zeros((self.N, 2), int)   # hall‑call counts
        self.waiting : List[Passenger] = []
        self.done    : List[Passenger] = []

    def _info(self):
        
        return {"time": self.time,
                "N": self.N,
                "M": self.M,
                "cars_itinerary": [c.itinerary for c in self.cars],
                "cars_direction": [c.direction for c in self.cars],
                "cars_position": [c.position for c in self.cars],
                "cars": self.cars,
                "hall_calls": self.hall}

    # ---------------- render --------------------------------------------
    def render(self, mode="human"):
        print(f"t={self.time:.1f}s | waiting={len(self.waiting)}")
        for k, c in enumerate(self.cars):
            print(f" Car{k} floor={c.position:.2f} dir={c.direction:+d}"
                  f" load={len(c.passengers)} itinerary={c.itinerary}")
        print(" Hall‑call floors:", np.where(self.hall.sum(1) > 0)[0].tolist())
        print("-"*40)
