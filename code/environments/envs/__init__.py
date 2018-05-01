from gym.envs.registration import register

register(
    id='MAB-iid-v0',
    entry_point='environments.envs.MAB.iid:MAB_iid')
register(
    id='MAB-random-walk-v0',
    entry_point='environments.envs.MAB.random_walk:MAB_random_walk')
register(
    id='PWEA-iid-v0',
    entry_point='environments.envs.PWEA.iid:PWEA_iid')
register(
    id='PWEA-iid-w-switch-v0',
    entry_point='environments.envs.PWEA.iid_w_switch:PWEA_iid_w_switch')
register(
    id='PWEA-markov-v0',
    entry_point='environments.envs.PWEA.markov:PWEA_markov')
