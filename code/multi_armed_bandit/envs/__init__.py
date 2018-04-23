from gym.envs.registration import register

register(
    id='MAB-iid-v0',
    entry_point='multi_armed_bandit.envs.iid:MAB_iid')
register(
    id='MAB-random-walk-v0',
    entry_point='multi_armed_bandit.envs.random_walk:MAB_random_walk')
