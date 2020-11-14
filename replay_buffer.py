import collections
import numpy as np
import copy

class ReplayBuffer():

    def __init__(self, buffer_size,sample=None):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=self.buffer_size)
        if sample != None:
            for i in range(sample[0].shape[0]):
                self.push(sample[0][i],sample[1][i],sample[2][i],
                        sample[3][i],sample[4][i])

    def push(self, state, action, reward, done, new_state, sub, overwrite=True):

        """ If maximum buffer size reached , remove old experience """
        if len(self.buffer) == self.buffer_size:
            self.buffer.popleft()

        state = copy.deepcopy(state)
        new_state = copy.deepcopy(new_state)
        action = copy.deepcopy(action)
        reward = copy.deepcopy(reward)
        new_state = copy.deepcopy(new_state)
        sub = copy.deepcopy(sub)

        """ Add new experience """
        self.buffer.append((state, action, reward, done, new_state, sub))

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):

        self.n = self.n + 1

        if self.n < len(self.buffer):
            return self.buffer[self.n]

        raise StopIteration

    def sample(self, batch_size=1):
        indices = np.random.randint(0, len(self.buffer), batch_size)
        experiences = [self.buffer[i] for i in indices]

        states = []
        actions = []
        rewards = []
        dones = []
        new_states = []
        sub = []

        for exp in experiences:
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            dones.append(exp[3])
            new_states.append(exp[4])
            sub.append(exp[5])


        np.array(states)
        np.array(actions)
        np.array(rewards)
        np.array(dones)
        np.array(new_states)
        np.array(sub)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(dones), np.array(new_states), np.array(sub))


    def clear(self):
        self.buffer.clear()
