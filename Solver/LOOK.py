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
        return idle_cars
    
    def _idle_action(self, calls, info, cars_pos):
        idle_car_idx = self._idle_car_idx(info)
        if idle_car_idx is None:
            return (info["N"], 0)
        
        # print(f"idle_car_idx: {idle_car_idx}")
        # print(f"calls: {calls}")
        lower_call = [0 for _ in range(len(idle_car_idx))]
        upper_call = [0 for _ in range(len(idle_car_idx))]
        call = [0 for _ in range(len(idle_car_idx))]
        direction = [0 for _ in range(len(idle_car_idx))]
        for i in range(len(idle_car_idx)):
            upper_call[i] = self._upper_call(calls[:,idle_car_idx[i], 0], cars_pos[idle_car_idx[i]])
            lower_call[i] = self._lower_call(calls[:,idle_car_idx[i], 1], cars_pos[idle_car_idx[i]])
            call[i] = self._call(calls[:,idle_car_idx[i], :])
            direction[i] = self._direction(info, idle_car_idx[i])
            
        # print(f"upper_call: {upper_call}, lower_call: {lower_call}, call: {call}, direction: {direction}")
        
        for i in range(len(idle_car_idx)):
            if direction[i] == 1:
                if upper_call[i] is not None:
                    return (upper_call[i], idle_car_idx[i])
                elif lower_call[i] is not None:
                    self.directions[idle_car_idx[i]] = -1
                    return (lower_call[i], idle_car_idx[i])
                elif call[i] is not None:
                    self.directions[idle_car_idx[i]] = np.sign(call[i] - cars_pos[idle_car_idx[i]])
                    return (call[i], idle_car_idx[i])
            elif direction[i] == -1:
                if lower_call[i] is not None:
                    return (lower_call[i], idle_car_idx[i])
                elif upper_call[i] is not None:
                    self.directions[idle_car_idx[i]] = 1
                    return (upper_call[i], idle_car_idx[i])
                elif call[i] is not None:
                    self.directions[idle_car_idx[i]] = np.sign(call[i] - cars_pos[idle_car_idx[i]])
                    return (call[i], idle_car_idx[i])
            elif direction[i] == 0:
                if call[i] is not None:
                    self.directions[idle_car_idx[i]] = np.sign(call[i] - cars_pos[idle_car_idx[i]])
                    return (call[i], idle_car_idx[i])
                
        
                

        
        return (info["N"], 0)

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
        
        cars_calls = np.concatenate([cars_calls, cars_calls], axis=2)
        cars_calls_action = self._idle_action(cars_calls, info, cars_pos)
        if cars_calls_action[0] != info["N"]:
            return cars_calls_action
        
        calls = cars_calls + hall_calls
        
        
        return self._idle_action(calls, info, cars_pos)
       
    def reset(self, info):
        self.directions = [0 for _ in range(info["M"])]

    def run_episode(self, max_steps=200):
        obs, info = self.env.reset()
        self.reset(info)
        
        total_reward = 0
        total_done = 0
        total_waiting = 0
        step_count = 0
        while True:
        
            step_count += 1
            action = self.get_next_action((obs, info))
            # print(f"action: {action}")
            obs, reward, done, truncated, info = self.env.step(action)
            
            total_reward += reward
            total_done = info["done"]
            total_waiting = info["waiting"]
            # env.render()
            if info["time"] > max_steps:
                break
            if done or truncated:
                break
        print(f"info[\"done\"]: {info['done']}")
        print(f"step_count: {step_count}")
        print(f"time: {info['time']}")
        return total_reward, total_done, total_waiting
    
    def benchmark(self, num_episodes=100):
        rewards = []
        for _ in range(num_episodes):
            total_reward, total_done, total_waiting = self.run_episode()
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
                   avg_passengers_spawning_time=5,
                   total_passengers=1000000,
                   capacity=3)
    solver = LOOKSolver(env)
    rewards = []
    rewards = solver.benchmark(num_episodes=100)
    # print(f"rewards: {rewards}")
    print(f"mean reward: {np.mean(rewards)}")
    solver.plot(rewards)