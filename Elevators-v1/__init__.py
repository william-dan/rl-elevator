from gymnasium.envs.registration import register

register(
    id="Elevators-v1/GridWorld-v0",
    entry_point="Elevators-v1.envs:GridWorldEnv",
)
