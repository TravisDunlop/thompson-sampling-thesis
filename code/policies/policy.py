class Policy:
    '''Generic Policy definition'''
    def __init__(self):
        pass

    def reset(self, env):
        self.action_space = env.action_space
        pass

    def get_name(self):
        pass

    def update(self, **kwargs):
        pass

    def act(self):
        pass

class Random(Policy):
    '''choose action at random'''
    def get_name(self):
        return 'random'

    def act(self, observation = None):
        return self.action_space.sample()

    def update(self, observation = None):
        pass
