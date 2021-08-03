import numpy as np


class InterpolateOriginal:
    @staticmethod
    def interpolate(x):
        return x


# TODO - methods to be implemented

# test function
if __name__ == "__main__":
    data = np.array([[1, 2, 3, 0, 5, 6, 7],
                     [0, 1, 2, 0, 4, 5, 0]])
    interpolator = InterpolateOriginal()
    data_filled = interpolator.interpolate(data)
    print('original:\n', data)
    print('interpolated:\n', data_filled)
