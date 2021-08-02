import numpy as np


class ContextGenerator:
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.remain = window_size % 2 != 0  # if window size is odd, remain sample point itself; else not
        self.k = window_size // 2  # half the window size

    def generate_contexts(self, data):
        contexts = None
        m, l = data.shape  # total samples; length
        # zero padding
        zero_vec = np.zeros((m, self.k))
        values = np.concatenate((zero_vec, data, zero_vec), axis=1)
        # sample each point --> contexts
        for idx in range(self.k, l + self.k, 1):
            # sample
            contexti = values[:, idx - self.k: idx]  # left
            if self.remain:  # point itself
                contexti = np.concatenate((contexti, values[:, idx: idx + 1]), axis=1)
            contexti = np.concatenate((contexti, values[:, idx + 1: idx + 1 + self.k]), axis=1)  # right
            # append
            contexti = contexti.reshape(contexti.shape[0], contexti.shape[1], 1)
            contexts = contexti if contexts is None else np.concatenate((contexts, contexti), axis=2)
        return np.array(contexts)


# test function
if __name__ == "__main__":
    generator = ContextGenerator(window_size=3)  # also try 4
    data = np.random.randn(2, 6)
    ctx = generator.generate_contexts(data)
