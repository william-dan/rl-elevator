# Elevator dispatch environment for Gym
This is a project for Reinforcement Learning Course in UniNE

## Directory Structure

- Elevators
  - envs
    - simple_elevator.py: contains the core logic of the environment
- Solver
  - BaseSolver.py: Base class for all the solvers
  - FIFO.py: implement FIFO to solve elevator dispatch
  - LOOK.py: implement LOOK to solve elevator dispatch 
- visualization: you can use `python *.py -h` for all the python files in this folder to check how to use it.
  - event_plot.py: parse the render output from this environment and plot a beautiful event timeline
  - load_plot.py: parse the render output from this environment and plot a beautiful event timeline, the darker the color is, the larger load elevator has. 
  - rewards_boxplot.py: You give a list of rewards to the program, it will produce a box plot to show the mean and variance of the rewards.
  - filling_plot.py: You give losses to it, it will produce a nice figure with running means and variance filling area.
  - ppo_handle.py: The PPO rewards are accumulated, I use this to recover the original rewards
- RL_elevator.ipynb: contains the core logic of training RL models




## Installation

To install your new environment, run the following commands:

```{shell}
pip install -r requirements.txt
cd Elevators
pip install -e .
```



## Contributing

If you would like to contribute, follow these steps:

- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).
