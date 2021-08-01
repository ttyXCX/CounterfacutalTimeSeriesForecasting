import numpy as np
from tsfresh.feature_extraction import feature_calculators as tsfc


class PhaseSpaceLearner:
    def __init__(self, output=False):
        self.functions = [tsfc.abs_energy,                     tsfc.absolute_sum_of_changes,
                          tsfc.benford_correlation,            tsfc.count_above_mean,
                          tsfc.count_below_mean,               tsfc.first_location_of_maximum,
                          tsfc.first_location_of_minimum,      tsfc.kurtosis,
                          tsfc.last_location_of_maximum,       tsfc.last_location_of_minimum,
                          tsfc.longest_strike_above_mean,      tsfc.longest_strike_below_mean,
                          tsfc.maximum,                        tsfc.mean,
                          tsfc.mean_abs_change,                tsfc.mean_change,
                          tsfc.mean_second_derivative_central, tsfc.median,
                          tsfc.minimum,                        tsfc.root_mean_square,
                          tsfc.skewness,                       tsfc.standard_deviation,
                          tsfc.variance,                       tsfc.variation_coefficient]
        self.output = output

    def learn_phase_space(self, series):
        m = series.shape[0]  # total samples
        phase_space = []
        for i in range(m):  # loop series
            if self.output:
                print('\rProcessing %d/%d...' % (i + 1, m), end='')
            si = series[i, :]
            phi = []
            for phfunc in self.functions:  # loop functions
                feature = phfunc(si)
                phi.append(feature)
            phase_space.append(phi)
        phase_space = np.array(phase_space)

        if self.output:
            print('Done. Phase space shape: %s' % str(phase_space.shape))
        return phase_space


if __name__ == "__main__":
    data = np.load('./datasets/ETT.npy', allow_pickle=True)
    ph_learner = PhaseSpaceLearner(output=True)
    ph = ph_learner.learn_phase_space(data)
    print(ph)
