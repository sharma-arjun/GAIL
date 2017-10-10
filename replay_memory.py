import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'phase', 'next_phase'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, next_state, reward, phase=0.0, next_phase=0.0):
        """Saves a transition."""
        self.memory.append(Transition(state, action, mask, next_state, reward, phase, next_phase))

    def sample(self):
        return Transition(*zip(*self.memory))

    def sample_batch(self, batch_size):
        random_batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)

class Memory_Ep(object):
    def __init__(self):
        self.memory = []

    def push(self, mem_obj):
        self.memory.append(mem_obj)

    def sample(self):
        return [Transition(*zip(*m.memory)) for m in self.memory]

    def sample_batch(self, batch_size):
        random_batch = random.sample(self.memory, batch_size)
        return [Transition(*zip(*m.memory)) for m in random_batch]
