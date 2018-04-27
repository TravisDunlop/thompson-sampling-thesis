from gym.envs.registration import register

register(
    id='MAB-iid-v0',
    entry_point='environments.envs.MAB.iid:MAB_iid')
register(
    id='MAB-random-walk-v0',
    entry_point='environments.envs.MAB.random_walk:MAB_random_walk')
register(
    id='PWEA-random-walk-v0',
    entry_point='environments.envs.PWEA.iid:PWEA_iid')
