import numpy as np
import pandas as pd
import os
import configparser


class DataFormatter:
    def __init__(self):
        self.input_length = None
        self.prediction_length = None
        self.header = None
        self.column = None
        self.ds_list = ['ETT', 'electricity', 'exchange', 'traffic', 'weather', 'ILI']

    # dataset should be in csv format
    def load_dataset(self, ds_path):
        ds = pd.read_csv(ds_path, header=self.header)
        ds = ds.values[:, self.column]

        print('Dataset loaded. Using column %d with shape %s'
              % (self.column, str(ds.shape)))
        return ds

    # load dataset configuration
    def load_config(self, ds_name, config_path):
        # check validation
        if ds_name not in self.ds_list:
            raise Exception('ERROR: Invalid dataset \'%s\'. Available datasets: %s'
                            % (str(ds_name), str(self.ds_list)))
        # load config
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(config_path)
        self.input_length = config.getint(ds_name, 'input_length')
        self.prediction_length = config.getint(ds_name, 'prediction_length')
        self.header = config.get(ds_name, 'header')
        self.header = int(self.header) if self.header is not None else None
        self.column = config.getint(ds_name, 'column')
        print('''Dataset \'%s\' configuration loaded.
        Input length = %d
        Prediction length = %d
        header = %s 
        column = %d''' % (str(ds_name),
                          self.input_length,
                          self.prediction_length,
                          str(self.header),
                          self.column))

    # slice raw dataset into samples
    def sample_series(self, series):
        series_length = series.shape[0]
        sample_length = self.input_length + self.prediction_length
        samples = None

        for idx in range(0, series_length - sample_length + 1, self.prediction_length):
            print('\rsampling series --> %d/%d...'
                  % (idx / self.prediction_length + 1, (series_length - self.input_length) / 12), end='')
            si = series[idx: idx + sample_length].reshape(1, sample_length)
            samples = si if samples is None else np.concatenate((samples, si), axis=0)

        print('Done. Data shape: %s' % (str(samples.shape)))
        return samples

    # an ensemble method
    def format_dataset(self, ds_name, ds_path, save_path, config_path='./DatasetConfig.ini'):
        # process data
        self.load_config(ds_name, config_path)
        data_series = self.load_dataset(ds_path)
        samples = self.sample_series(data_series)
        # check path validity
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save
        np.save(save_path + '\%s.npy' % ds_name, samples)
        print("Formatted dataset saved at: \'%s\%s.npy'" % (str(save_path), str(ds_name)))
        return samples


# test function
if __name__ == "__main__":
    ds_path = r".\datasets\raw\ILI\ILINet.csv"
    ds_name = 'ILI'
    save_path = r".\datasets"

    formatter = DataFormatter()
    samples = formatter.format_dataset(ds_name, ds_path, save_path)
    print(samples)
