from gym.envs.registration import register

register(
    id='ver-v0',
    entry_point='gym_ver.envs:VerEnv',
)
