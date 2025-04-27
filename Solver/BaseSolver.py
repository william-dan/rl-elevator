from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """
    Abstract base class for solver implementations such as FIFO and LOOK.
    """

    @abstractmethod
    def __init__(self, env):
        """
        Initialize the solver with the environment.
        """
        pass

    @abstractmethod
    def get_next_action(self, current_state):
        """
        Return the next action based on the current state.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the solver to its initial state.
        """
        pass
    
    @abstractmethod
    def run_episode(self, max_steps=100):
        """
        Run an episode of the environment using the solver.
        """
        pass
    
    @abstractmethod
    def plot(self, rewards):
        """
        Plot the performance of the solver.
        """
        pass
    
    @abstractmethod
    def benchmark(self, num_episodes=100):
        """
        Benchmark the solver's performance over a number of episodes.
        """
        pass