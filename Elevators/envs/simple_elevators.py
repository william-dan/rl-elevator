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
from enum import Enum

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

class Event(Enum):
    CAR_DOOR_CLOSE = 0
    CAR_DOOR_OPEN  = 1
    SPAWN_PASSENGER = 2


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
                 avg_passengers_spawning_time: float = 50.0,
                 max_passengers_at_a_time: int = 10,
                 avg_passengers_at_a_time: float = 5.0,
                 seed: int | None = None):

        self.N, self.M = num_floors, num_cars
        self.cap       = capacity
        self.v         = speed_m_s
        self.sec_floor = floor_h_m / speed_m_s
        # print(f"sec_floor: {self.sec_floor}")
        self.t_door    = door_time
        # self.lambda_p  = passenger_rate
        self.avg_passengers_spawning_time = avg_passengers_spawning_time
        self.rng       = np.random.default_rng(seed)
        self.max_passengers_at_a_time = max_passengers_at_a_time
        self.avg_passengers_at_a_time = avg_passengers_at_a_time
        # observation = (N × M × 5) tensor flattened
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.N, self.M, 5,), dtype=np.float32)
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

        reward, event = self._advance_until_event()      # discrete‑event simulation
        self.current_event = event
        return self._obs(), reward, False, False, self._info()

    # ---------------- simulation core -----------------------------------
    
    def _advance_until_event(self) -> float:
        """Run until next passenger or door‑closure event, return reward."""
        self.new_request_this_step = None
        self.current_event = None
        
        # ––– time until next Poisson arrival –––
        if self.t_pass is None:
            self.t_pass = self.rng.exponential(self.avg_passengers_spawning_time) + self.time
        

        # ––– time until any car event –––
        t_car = np.inf
        for car in self.cars:
            if car.door_open:
                t_car = min(t_car, car.t_door)
            elif car.itinerary is not None:
                t_car = min(t_car,
                            abs(car.itinerary - car.position) * self.sec_floor)

        dt = min(self.t_pass, t_car + self.time) - self.time
        self.time += dt

        is_end = False
        end_reason = None
        # move / count down doors
        for car in self.cars:
            if car.door_open:
                car.t_door -= dt
                if car.t_door <= 1e-6:
                    car.door_open, car.t_door = False, 0.0
                    # assert is_end is False
                    is_end = True
                    end_reason = Event.CAR_DOOR_CLOSE
            elif car.itinerary is not None:
                sign = np.sign(car.itinerary - car.position)
                car.position += sign * (dt / self.sec_floor)
                car.direction = int(sign)
                if abs(car.position - car.itinerary) < 1e-3:
                    car.position = float(car.itinerary)
                    car.itinerary = None
                    self._handle_arrival(car, car.direction)         
                    car.door_open, car.t_door = True, self.t_door
                    # assert is_end is False
                    is_end = True
                    end_reason = Event.CAR_DOOR_OPEN
            else:
                car.direction = 0

        # passenger arrival occurs first
        if abs(self.t_pass - self.time) < 1e-3:
            self._spawn_passenger()
            self.t_pass = None
            # assert is_end is False
            is_end = True
            end_reason = Event.SPAWN_PASSENGER

        assert is_end is True, "is_end should be True"
        return self._reward_snapshot(), end_reason

    # ---------------- boarding/alighting & generator ---------------------
    def _handle_arrival(self, car: Car, prev_direction: int):
        # print(f"car arrived at floor {car.position}")
        floor = int(car.position)
        # alight
        for p in list(car.passengers):
            if p.destination == floor:
                p.t_alight = self.time
                self.done.append(p)
                car.passengers.remove(p)
        # board
        
        if prev_direction == 0:  # idle
            prev_direction = self.rng.choice([-1, 1])
        waiting_here = [p for p in self.waiting if p.origin == floor and prev_direction * (p.destination - floor) > 0]
        car.direction = 0
        # space = self.cap - len(car.passengers) #!!!!!!!!!!!!!
        # for p in waiting_here[:space]:
        for p in waiting_here:
            p.t_board = self.time
            car.passengers.append(p)
            self.waiting.remove(p)
            # if p.destination not in car.itinerary:
            #     car.itinerary.append(p.destination)
        self.hall[floor, 0 if prev_direction == 1 else 1] = 0  #! if capacity is limited, should not be like this
        # print(f"hall_calls: {self.hall}")
        
    def _spawn_passenger(self):
        n_passengers = self.rng.exponential(scale=self.avg_passengers_at_a_time) + 1
        o = self.rng.integers(0, self.N)
        for i in range(int(n_passengers)):
            d = self.rng.choice([f for f in range(self.N) if f != o])
            self.waiting.append(Passenger(o, d, self.time))
            self.new_request_this_step = o
            self.hall[o, 0 if d > o else 1] |= 1

    # ---------------- reward --------------------------------------------
    def _reward_snapshot(self): 
        w = sum((self.time - p.t_request) for p in self.waiting)
        r = sum((self.time - p.t_board)
                for car in self.cars for p in car.passengers if p.t_board)
        return -(w + r) * 1e-3 

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
        # print(f"cars: {[c.position for c in self.cars]}")
        P = np.repeat(np.array([c.position for c in self.cars]).reshape(len(self.cars), 1), self.N, 1).T
        D = np.repeat(np.array([c.direction for c in self.cars]).reshape(len(self.cars), 1), self.N, 1).T
        # print(f"P shape: {P.shape}, D shape: {D.shape}")
        # print(f"B shape: {B.shape}, A shape: {A.shape}")
        stacked = np.stack([B[:, j, :] for j in range(2)] + [A, P, D], 2)
        # print(f"stacked shape: {stacked.shape}")
        return stacked.astype(np.float32)

    def _reset_state(self):
        self.time = 0.0
        self.cars    = [Car(self.cap) for _ in range(self.M)]
        self.hall    = np.zeros((self.N, 2), int)   # hall‑call counts
        self.waiting : List[Passenger] = []
        self.done    : List[Passenger] = []
        self.current_event = None
        self.t_pass = None

    def _info(self):
        
        return {"time": self.time,
                "N": self.N,
                "M": self.M,
                "cars_itinerary": [c.itinerary for c in self.cars],
                "cars_direction": [c.direction for c in self.cars],
                "cars_position": [c.position for c in self.cars],
                "cars": self.cars,
                "hall_calls": self.hall,
                "idle_cars": self._find_idle_cars(),
                "waiting": len(self.waiting),
                "done": len(self.done),
                }
    
    # ---------------- utils -------------------------------------
    def _find_new_requests(self):
        return self.new_request_this_step
    
    def _find_idle_cars(self):
        idle_cars = []
        for i, car in enumerate(self.cars):
            if car.itinerary is None:
                idle_cars.append(i)
        return idle_cars
    
    def _find_cars_with_itinerary(self):
        cars_with_itinerary = []
        for i, car in enumerate(self.cars):
            if car.itinerary is not None:
                cars_with_itinerary.append(i)
        return cars_with_itinerary

    def _find_cars_with_passengers(self):
        cars_with_passengers = []
        for i, car in enumerate(self.cars):
            if len(car.passengers) > 0:
                cars_with_passengers.append(i)
        return cars_with_passengers


    # ---------------- render --------------------------------------------
    def render(self, mode="human"):
        event_to_str = {
            Event.CAR_DOOR_CLOSE: "CLOSE",
            Event.CAR_DOOR_OPEN:  "OPEN",
            Event.SPAWN_PASSENGER: "SPAWN"
        }
        print(f"t={self.time:.1f}s | waiting={len(self.waiting)}")
        color_map = {
            Event.CAR_DOOR_CLOSE: "\033[91m",  # Red
            Event.CAR_DOOR_OPEN:  "\033[92m",  # Green
            Event.SPAWN_PASSENGER: "\033[93m"  # Yellow
        }
        color_prefix = color_map.get(self.current_event, "\033[0m")
        print(f"{color_prefix}current event: {event_to_str[self.current_event]}\033[0m")
        for k, c in enumerate(self.cars):
            print(f" Car{k} floor={c.position:.2f} dir={c.direction:+d}"
                  f" load={len(c.passengers)} itinerary={c.itinerary}")
        print(" Hall‑call floors:", np.where(self.hall.sum(1) > 0)[0].tolist())
        print("-"*40)
