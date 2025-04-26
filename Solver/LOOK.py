import gymnasium as gym
import Elevators
from collections import deque

import Elevators.envs
import Elevators.envs.simple_elevators

class LOOKSolver:
    def __init__(self, env: gym.Env):
        self.env = env
        

    
    def _idle_action(self, calls, info, cars_pos):
        # print(f"calls: {calls}")
        # print(f"cars_pos: {cars_pos}")
        idle_cars: list[int] = info["idle_cars"]
        if len(idle_cars) == 0:
            return (info["N"], 0)
        assert len(idle_cars) > 0, "idle cars must be > 0"
        idle_car_idx = idle_cars[0]
        if self.directions[idle_car_idx] == 1:
            # print(cars_pos)
            # cars_pos[idle_car_idx]
            upper_calls = calls[:,idle_car_idx, 0]
            if upper_calls.sum() > 0:
                for i in range(cars_pos[idle_car_idx], len(upper_calls)):
                    if upper_calls[i] > 0:
                        return (i, idle_car_idx)
            else:
                self.directions[idle_car_idx] = -1
                lower_calls = calls[:,idle_car_idx, 1]
                if lower_calls.sum() > 0:
                    for i in range(cars_pos[idle_car_idx], -1, -1):
                        if lower_calls[i] > 0:
                            return (i, idle_car_idx)
                    
        else:
            lower_calls = calls[:,idle_car_idx, 1]
            if lower_calls.sum() > 0:
                for i in range(cars_pos[idle_car_idx], -1, -1):
                    if lower_calls[i] > 0:
                        return (i, idle_car_idx)
                
            else:
                self.directions[idle_car_idx] = 1
                upper_calls = calls[:,idle_car_idx, 0]
                if upper_calls.sum() > 0:
                    for i in range(cars_pos[idle_car_idx], len(upper_calls)):
                        if upper_calls[i] > 0:
                            return (i, idle_car_idx)
        return (info["N"], 0)  # No action if no requests

    def select_action(self, obs, info):
        N = info["N"]
        M = info["M"]
        event = info["event"]
        itinerary = info["cars_itinerary"]
    
        cars_calls = obs[:,:,2].reshape(N, M, 1)
        hall_calls = obs[:,:,0:2]
        for i in range(M):
            if itinerary[i] is not None:
                print(f"itinerary[i]: {itinerary[i]}")
                hall_calls[itinerary[i],:,:] = 0
            
        cars_pos = obs[0,:,3].astype(int)
        
        
        calls = cars_calls + hall_calls
        
        if event == Elevators.envs.simple_elevators.Event.CAR_DOOR_OPEN:
            return (info["N"], 0)  # No action if no requests
        elif event == Elevators.envs.simple_elevators.Event.CAR_DOOR_CLOSE:
            return self._idle_action(calls, info, cars_pos)
            
        elif event == Elevators.envs.simple_elevators.Event.SPAWN_PASSENGER:
            return self._idle_action(calls, info, cars_pos)       
            
        return (info["N"], 0)  # No action if no requests
        

    def run_episode(self, max_steps=20):
        obs, info = self.env.reset()
        self.directions = [1 for _ in range(info["M"])]
        
        
        total_reward = 0
        for _ in range(max_steps):
            action = self.select_action(obs, info)
            print(f"action: {action}")
            obs, reward, done, truncated, info = self.env.step(action)
            
            total_reward += reward
            env.render()
            if done or truncated:
                break
        print(f"info[\"done\"]: {info['done']}")
        return total_reward
    
    def plot(self, rewards):
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('LOOKSolver Performance')
        plt.show()

if __name__ == "__main__":
    env = gym.make("Elevators/Elevators-v0", num_floors=20, num_cars=4)
    solver = LOOKSolver(env)
    rewards = []
    for _ in range(1):
        total_reward = solver.run_episode(max_steps=20)
        rewards.append(total_reward)
    solver.plot(rewards)