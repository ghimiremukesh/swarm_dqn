from gym.envs.registration import register

register(
    id="swarm-v0",
    entry_point="swarm.envs:SwarmEnv"
)