import gymnasium as gym
import Elevators

env = gym.make("Elevators/Elevators-v0", num_floors=20, num_cars=4, passenger_rate=0.3)
obs, info = env.reset(seed=0)
for _ in range(200):
    action = env.action_space.sample()   # random dispatch
    obs, reward, terminated, truncated, info = env.step(action)
    if _ % 20 == 0: env.render()
