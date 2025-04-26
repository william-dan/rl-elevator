import gymnasium as gym

class LOOKSolver:
    """
    A simple solver that implements the LOOK algorithm for an ElevatorEnv.
    Keeps track of floors where requests exist and moves in the current direction
    until there are no more requests, then reverses direction.
    """
    def __init__(self, env: gym.Env):
        self.env = env
        

    def select_action(self, info):
        """
        Selects the next action based on the LOOK algorithm.
        eg:
        num_floors = 10
        num_cars = 4
        hall_calls = [2, 4, 6, 9]
        cars_position = [0, 1, 2, 3]
        """
        
        
        hall_calls = info['hall_calls']
        self.upper_requests = [i for i in range(len(hall_calls)) if hall_calls[i][0] > 0]
        self.lower_requests = [i for i in range(len(hall_calls)) if hall_calls[i][1] > 0]
        if not 
        
        self.request_direction = 1 if self.upper_requests else -1
        
        current_floor = info["cars_position"]

        if not new_requests:
            return self.env.action_space.sample()  # No action if no requests

        # Determine if we still have requests in current direction
        if self.direction > 0:
            upper_requests = [r for r in requests if r >= current_floor]
            if upper_requests:
                return min(upper_requests, key=lambda r: abs(r - current_floor))
            else:
                # Reverse direction if no requests above
                self.direction = -1
        else:
            lower_requests = [r for r in requests if r <= current_floor]
            if lower_requests:
                return max(lower_requests, key=lambda r: abs(r - current_floor))
            else:
                # Reverse direction if no requests below
                self.direction = 1

        # If direction reversed, compute new floor choice
        if self.direction > 0:
            return min(requests, key=lambda r: abs(r - current_floor))
        else:
            return max(requests, key=lambda r: abs(r - current_floor))

    def run_episode(self, max_steps=1000):
        done = False
        observation, info = self.env.reset()
        self.prev_hall_calls = info['hall_calls']
        total_reward = 0

        while not done and max_steps > 0:
            max_steps -= 1
            
            action = self.select_action(info)
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

        return total_reward