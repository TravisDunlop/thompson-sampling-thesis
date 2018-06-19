'''In this file I implement the A-star algorithm for identifying
   adversarial examples.  It's guaranteed to find the combination of losses which
   maximize expected cummulative regret against an algorithm.  In  this case
   we implement it for Thompson Sampling.'''

from math import inf
from queue import PriorityQueue
from itertools import product
from collections import Counter

import numpy as np
from numpy import argmax, dot
from numpy.random import beta, randint

############################################################
# helper functions

def permute(num_options, num_items):
    '''generate a list of tuples with all possible combinations of the num_options
        for a length num_items.  This is used to initialize all starting values'''
    return list(product(list(range(num_options)), repeat = num_items))

def summarize_actions(actions, num_estimations, num_actions):
    '''take a vector of actions and create a list showing the percentage of times
        the action was taken.  In this case, an action is a value between 0 and
        num_actions - 1 corresponding to the lever chosen.'''
    counter = Counter(actions)
    action_pct = [counter[i] / num_estimations for i in range(num_actions)]
    return action_pct

def subtract_tuple(a, b):
    return tuple(i - j for i, j in zip(a, b))

def is_admissable_tuple(t, min, max):
    '''check if a tuples values are between or equal to the min and max values'''
    for i, item in enumerate(t):
        if item < min[i] or item > max[i]:
            return False
    return True

def get_admissable_losses(cum_loss, num_timesteps, possible_tuples):
    min = tuple(i - num_timesteps + 1 for i in cum_loss)
    max = cum_loss
    admissable_losses = [p for p in possible_tuples if is_admissable_tuple(p, min, max)]
    return admissable_losses

############################################################

############################################################
# Node Class

class EndNode:
    ''' EndNode is the artificial final node in the graph of which we find the shortest path'''
    def __init__(self, num_actions, num_timesteps, num_estimations):
        self.future_regret = 0
        self.present_regret = 0
        self.best_action = None
        self.num_actions = num_actions
        self.num_timesteps = num_timesteps
        self.timestep = num_timesteps + 1
        self.num_estimations = num_estimations

class StartNode:
    ''' StartNode is the artificial first node in the graph from which we find the shortest path'''
    def __init__(self, parent):
        self.future_regret = parent.future_regret + parent.present_regret
        self.present_regret = 0

    def is_goal_node(self):
        return True


class Node:
    def __init__(self, parent, cum_loss, present_loss, timestep = None):

        self.cum_loss = cum_loss
        self.present_loss = present_loss

        self.parent = parent
        self.num_estimations = parent.num_estimations
        self.num_actions = parent.num_actions
        self.num_timesteps = parent.num_timesteps
        if parent.best_action is None: self.best_action = argmax(cum_loss)
        else: self.best_action = parent.best_action

        if timestep is None:
            self.timestep = parent.timestep - 1
        else:
            self.timestep = timestep

        self.present_regret = self.estimate_present_regret()
        self.future_regret = parent.future_regret + parent.present_regret
        self.heuristic_past_regret = self.heuristic_regret_to_goal()

        self.estimated_cost = - self.present_regret - self.future_regret - self.heuristic_past_regret

    def __eq__(self, other):
        '''overriding equals operator to be able to compare two Nodes'''
        if self.cum_loss == other.cum_loss and \
            self.present_loss == other.present_loss and \
            self.best_action == other.best_action and \
            self.timestep == other.timestep and \
            self.num_actions == other.num_actions and \
            self.num_timesteps == other.num_timesteps:
            return True
        else:
           return False

    def __lt__(self, other):
        '''hack to make nodes comparable to one another for PriorityQueue.
            lt means less than.'''
        return True

    def __repr__(self):
        '''changing what's printed when the object is displayed'''
        repr = 'cum_loss : {}\n'.format(', '.join(str(i) for i in self.cum_loss))
        repr += 'present_loss : {}\n'.format(', '.join(str(i) for i in self.present_loss))
        repr += 'timestep : {}\n'.format(self.timestep)
        repr += 'present_regret : {}\n'.format(self.present_regret)

        return repr

    def copy(self, other):
        '''copy parent and important parameters of another Node'''
        self.parent = other.parent
        self.present_regret = other.present_regret
        self.future_regret = other.future_regret
        self.heuristic_past_regret = other.heuristic_past_regret
        self.estimated_cost = other.estimated_cost

    def get_children(self):
        '''get all admissible children from this node'''
        if self.timestep == 1:
            return [StartNode(self)]

        possible_tuples = permute(2, num_actions)

        cum_loss = subtract_tuple(self.cum_loss, self.present_loss)

        present_losses = get_admissable_losses(cum_loss, self.timestep - 1, possible_tuples)

        children = [Node(self, cum_loss, present_loss) for present_loss in present_losses]

        return children


    def estimate_present_regret(self):
        '''estimate instantaneous regret for this time step'''

        #parameters of thompson sampling
        a = [1 + i for i in self.cum_loss]
        b = [1 + self.timestep - i for i in self.cum_loss]

        # sampling num_estimations actions
        actions = [beta(a, b).argmin() for _ in range(num_estimations)]
        action_pct = summarize_actions(actions, num_estimations, num_actions)

        regret = dot(self.present_loss, action_pct) - self.present_loss[self.best_action]

        return regret

    def is_goal_node(self):
        return False

    def heuristic_regret_to_goal(self):
        return self.timestep - self.cum_loss[self.best_action]

############################################################

############################################################
# helper functions

def get_starting_nodes(num_actions, num_timesteps, num_estimations):
    cum_losses = permute(num_timesteps, num_actions)

    possible_tuples = permute(2, num_actions)

    end_node = EndNode(num_actions, num_timesteps, num_estimations)

    starting_nodes  = []

    for cum_loss in cum_losses:
        present_losses = get_admissable_losses(cum_loss, num_timesteps, possible_tuples)

        for present_loss in present_losses:
            starting_nodes.append(Node(end_node, cum_loss, present_loss))

    return starting_nodes

def put_node(priority_queue, node):
    priority_queue.put((node.estimated_cost, node))

def update_node(node, open_nodes):
    '''updates child if exists in list.  Returns true if the node was updated'''
    for _, n in open_nodes.queue:
        if n == node:
            if n.estimated_cost > node.estimated_cost:
                n.copy(node)
            return True
    return False


############################################################
## Testing Node class

num_actions = 3
num_timesteps = 5
num_estimations = 100

parent = EndNode(num_actions, num_timesteps, num_estimations)
cum_loss = (5, 4, 1)
present_loss = (1, 1, 0)
node = Node(parent, cum_loss, present_loss)

children = node.get_children()

print(node)

for child in children:
    print(child)

starting_nodes = get_starting_nodes(num_actions, num_timesteps, num_estimations)

for n in starting_nodes: put_node(open_nodes, n)


############################################################

num_actions = 3
num_timesteps = 5
num_estimations = 100

starting_nodes = get_starting_nodes(num_actions, num_timesteps, num_estimations)

open_nodes = PriorityQueue()
closed_nodes = []

for n in starting_nodes: put_node(open_nodes, n)

while not open_nodes.empty():

    _, node = open_nodes.get()

    if node.is_goal_node():
        result = reconstruct_path(node)
        break
    closed_nodes.append(node)

    children = node.get_children()
    for child in children:
        if child in closed_nodes:
            continue

        if not update_node(child, open_nodes):
            put_node(open_nodes, child)

children

def reconstruct_path(node):
    loss = np.zeros((node.num_actions, node.num_timesteps))
    while not isinstance(node, EndNode):
        loss[:, node.timestep - 1] = node.present_loss
        node = node.parent
    return loss

node.future_regret

open_nodes.queue[0:5]


result = reconstruct_path(node)

children

closed_nodes

child

child in [n for _, n in open_nodes.queue]

len(open_nodes.queue)

len(closed_nodes)

node


class apple:
    def __eq__(self, other):
        return True

class orange:
    def __eq__(self, other):
        return False

a = apple()
o = orange()

o == a
