import gymnasium as gym
from collections import deque
from BaseSolver import BaseSolver
import matplotlib.pyplot as plt
from tqdm import tqdm

class FIFOSolver(BaseSolver):
    def __init__(self, env: gym.Env):
        self.env = env
        self.queue = deque()

    def reset(self):
        self.queue.clear()

    def get_next_action(self, info):
        cars_itinerary = info["cars_itinerary"]
        
        
                
                
        diff = info['hall_calls'] - self.prev_hall_calls
        self.prev_hall_calls = info['hall_calls'].copy()
        
        
        completed_requests = [i for i in range(len(diff)) if diff[i][0] < 0 or diff[i][1] < 0]
        for i in completed_requests:
            if i in self.queue:
                self.queue.remove(i)
        
        new_requests = [i for i in range(len(diff)) if diff[i][0] > 0]
        new_requests.extend([i for i in range(len(diff)) if diff[i][1] > 0])
        # assert len(new_requests) == 0 or len(new_requests) == 1, "FIFO only supports one request at a time"
        self.queue.extend(new_requests)
        
        for i in range(len(cars_itinerary)):
            if cars_itinerary[i] is None:
                car_passengers = info["cars"][i].passengers
                if car_passengers:
                    return (car_passengers[0].destination, i)
        
        # print(f"queue: {self.queue}")
        if len(self.queue) > 0 and (None in cars_itinerary):
            target = self.queue[0]
            target_car = [i for i in range(len(cars_itinerary)) if cars_itinerary[i] is None]
            return (target, target_car[0])
        
        
        return (info["N"], 0)  # No action if no requests
        

    def run_episode(self, max_steps=10000000):
        obs, info = self.env.reset()
        self.reset()
        self.prev_hall_calls = info['hall_calls'].copy()
        total_reward = 0
        for _ in (range(max_steps)):
            
            action = self.get_next_action(info)
            obs, reward, done, truncated, info = self.env.step(action)
            # self.env.render()
            total_reward += reward
            if done or truncated:
                break
        # total_reward /= info["time"]
        # self.env.render()
        
        # print(f"info[\"done\"]: {info['done']}")
        return total_reward
    
    def plot(self, rewards):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('FIFOSolver Performance')
        plt.savefig("fifo_performance.png")
        plt.show()
        
        
        
    def benchmark(self, num_episodes=100):
        rewards = []
        for _ in range(num_episodes):
            total_reward = self.run_episode()
            rewards.append(total_reward)
        return rewards

if __name__ == "__main__":
    import Elevators
    env = gym.make("Elevators/Elevators-v0", 
                   num_floors=20, 
                   num_cars=4, 
                   avg_passengers_spawning_time=5,
                   total_passengers=50,
                   capacity=3)
    solver = FIFOSolver(env)
    rewards = solver.benchmark(num_episodes=10)
    print(f"Average reward: {sum(rewards) / len(rewards)}")
    solver.plot(rewards)