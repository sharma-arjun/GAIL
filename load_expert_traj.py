import numpy as np
from collections import namedtuple
import os
from running_state import ZFilter

Trajectory = namedtuple('Trajectory', ('state', 'action', 'phase', 'mask'))

class Expert(object):
    def __init__(self, folder, num_inputs):
        self.memory = []
        self.pointer = 0
        self.n = len(os.listdir(folder))
        self.folder = folder
        self.running_state = ZFilter((num_inputs,), clip=5)

    def push(self):
        """Saves a (state, action, phase, mask) tuple."""
        for i in range(self.n):
            f = open(self.folder + str(i) + '.txt', 'r')
            line_counter = 0
            temp_mem = []
            for line in f:
                if line_counter % 3 == 0:
                    if line_counter > 0:
                        temp_mem.append(Trajectory(s, a, phase, 1))
                    #s = self.running_state(np.asarray(line.strip().split(), dtype='float'))
                    s = np.asarray(line.strip().split(), dtype='float')
                elif line_counter % 3 == 1:
                    a = np.asarray(line.strip().split(), dtype='float')
                elif line_counter % 3 == 2:
                    phase = float(line.strip())

                line_counter += 1

            f.close()
            temp_mem.append(Trajectory(s, a, phase, 0))
            self.memory.append(Trajectory(*zip(*temp_mem)))


    def sample(self, size=5):
        ind = np.random.randint(self.n, size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        return Trajectory(*zip(*batch_list))

    #def sample_batch(self, batch_size):
    #    random_batch = random.sample(self.memory, batch_size)
    #    return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)
