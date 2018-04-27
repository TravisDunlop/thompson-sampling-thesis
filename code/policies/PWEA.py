from .policy import Policy

class Exponential(Policy):
    '''Exponential Weights with worst case learning rates as described in
        Prediction, Learning, and Games by Cesa-Bianchi, Lugosi 2006 Sections
        2.2 and 2.3'''
    def __init__(self, learning_rate = 'worst-case'):
        self.learning_rate = learning_rate

    def reset(self, env):
        if self.learning_rate == 'worst-case':
            pass
        else:
            raise Exception('learning_rate parameter not recognized')

    def get_name(self):
        return 'exponential weights - ' + self.learning_rate

    def update(self, action, reward):
        pass

    def act(self):
        pass
