class Policy:
    '''Generic Policy definition'''
    def __init__(self):
        pass

    def reset(self, env):
        self.action_space = env.action_space
        pass

    def get_name(self):
        pass

    def update(self, action, reward):
        pass

    def act(self):
        pass

class Random(Policy):
    '''choose action at random'''
    def get_name(self):
        return 'random'

    def act(self):
        return self.action_space.sample()
