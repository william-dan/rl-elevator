from gymnasium.envs.registration import register

register(
    id="Elevators/Elevators-v0",
    entry_point="Elevators.envs:ElevatorEnv",
)
