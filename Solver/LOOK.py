import gymnasium as gym
import Elevators
from collections import deque
import numpy as np

import Elevators.envs
import Elevators.envs.simple_elevators

class LOOKSolver:
    def __init__(self, env: gym.Env):
        self.env = env
        
    def _upper_call(self, upper_calls_per_car, car_pos):
        for i in range(car_pos+1, len(upper_calls_per_car)):
            if upper_calls_per_car[i] > 0:
                return i
        return None
    
    def _lower_call(self, lower_calls_per_car, car_pos):
        for i in range(car_pos-1, -1, -1):
            if lower_calls_per_car[i] > 0:
                return i
        return None

    def _call(self, calls):
        for i in range(len(calls)):
            if calls[i][0] > 0 or calls[i][1] > 0:
                return i
        return None
    
    def _direction(self, info, car_idx):
        
        return info["cars"][car_idx].direction

    def _idle_car_idx(self, info):
        idle_cars: list[int] = info["idle_cars"]
        if len(idle_cars) == 0:
            return None
        return idle_cars[0]
    
    def _idle_action(self, calls, info, cars_pos):
        idle_car_idx = self._idle_car_idx(info)
        if idle_car_idx is None:
            return (info["N"], 0)
        
        # print(f"calls: {calls}")
        upper_call = self._upper_call(calls[:,idle_car_idx, 0], cars_pos[idle_car_idx])
        lower_call = self._lower_call(calls[:,idle_car_idx, 1], cars_pos[idle_car_idx])
        call = self._call(calls[:,idle_car_idx, :])
        direction = self.directions[idle_car_idx]
        # print(f"upper_call: {upper_call}, lower_call: {lower_call}, call: {call}, direction: {direction}")
        
        
        if direction == 1:
            
            if upper_call is not None:
                return (upper_call, idle_car_idx)
            elif lower_call is not None:
                self.directions[idle_car_idx] = -1
                return (lower_call, idle_car_idx)
            elif call is not None:
                self.directions[idle_car_idx] = np.sign(call - cars_pos[idle_car_idx])
                return (call, idle_car_idx)
            else:
                return (info["N"], 0)
                    
        elif direction == -1:
            if lower_call is not None:
                return (lower_call, idle_car_idx)
            elif upper_call is not None:
                self.directions[idle_car_idx] = 1
                return (upper_call, idle_car_idx)
            elif call is not None:
                self.directions[idle_car_idx] = np.sign(call - cars_pos[idle_car_idx])
                return (call, idle_car_idx)
            else:
                return (info["N"], 0)
        
        elif direction == 0:
            if call is not None:
                self.directions[idle_car_idx] = np.sign(call - cars_pos[idle_car_idx])
                return (call, idle_car_idx)
            else:
                return (info["N"], 0)

        else:
            raise ValueError(f"Invalid direction: {direction}")
    

    def get_next_action(self, current_state):
        obs, info = current_state
        N = info["N"]
        M = info["M"]
        itinerary = info["cars_itinerary"]
    
        cars_calls = obs[:,:,2].reshape(N, M, 1)
        hall_calls = obs[:,:,0:2]
        for i in range(M):
            if itinerary[i] is not None:
                # print(f"itinerary[i]: {itinerary[i]}")
                hall_calls[itinerary[i],:,:] = 0
            
        cars_pos = obs[0,:,3].astype(int)
        
        calls = cars_calls + hall_calls
        
        return self._idle_action(calls, info, cars_pos)
       
        

    def run_episode(self, max_steps=20):
        obs, info = self.env.reset()
        self.directions = [0 for _ in range(info["M"])]
        
        
        total_reward = 0
        total_done = 0
        total_waiting = 0
        for _ in range(max_steps):
            action = self.get_next_action((obs, info))
            # print(f"action: {action}")
            obs, reward, done, truncated, info = self.env.step(action)
            
            total_reward += reward
            total_done = info["done"]
            total_waiting = info["waiting"]
            # env.render()
            if done or truncated:
                break
        # print(f"info[\"done\"]: {info['done']}")
        return total_reward, total_done, total_waiting
    
    def benchmark(self, num_episodes=100):
        rewards = []
        for _ in range(num_episodes):
            total_reward, total_done, total_waiting = self.run_episode(max_steps=100)
            rewards.append(total_reward)
        return rewards
    
    
    
    def plot(self, rewards):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('LOOKSolver Performance')
        plt.savefig("look_performance.png")
        plt.show()
        

if __name__ == "__main__":
    env = gym.make("Elevators/Elevators-v0", 
                   num_floors=20, 
                   num_cars=4, 
                   avg_passengers_spawning_time=5)
    solver = LOOKSolver(env)
    rewards = []
    rewards = solver.benchmark(num_episodes=100)
    print(f"mean reward: {np.mean(rewards)}")
    solver.plot(rewards)