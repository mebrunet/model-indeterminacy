import os

from .interface import TabularData


class SafeDriverData(TabularData):
    def __init__(self, datapath):

        self._datapath = datapath
        train_df, test_df = self.load_dataset(os.path.join(datapath, 'safe_driver'))

        self._train_df = train_df
        self._test_df = test_df
        self._target = 'target'

    @property
    def binaries(self):
        return ['ind_06_bin',
                'ind_07_bin',
                'ind_08_bin',
                'ind_09_bin',
                'ind_10_bin',
                'ind_11_bin',
                'ind_12_bin',
                'ind_13_bin',
                'ind_16_bin',
                'ind_17_bin',
                'ind_18_bin']

    @property
    def categoricals(self):
        return ['ind_02_cat',
                'ind_04_cat',
                'ind_05_cat',
                'car_01_cat',
                'car_02_cat',
                'car_03_cat',
                'car_04_cat',
                'car_05_cat',
                'car_06_cat',
                'car_07_cat',
                'car_08_cat',
                'car_09_cat',
                'car_10_cat']

    @property
    def ordinals(self):
        return ['ind_01',
                'ind_03',
                'ind_14',
                'ind_15']

    @property
    def continuous(self):
        return ['reg_01',
                'reg_02',
                'car_11',
                'car_12',
                'car_13',
                'car_15']

    @property
    def immutables(self):
        return ['reg_01',
                'reg_02',
                'ind_01',  # assume 1/3 of individual variables are immutable
                'ind_02_cat',
                'ind_03',
                'ind_04_cat',
                'ind_05_cat',
                'ind_06_bin']

    @property
    def target(self):
        return self._target

    @property
    def train(self):
        return self._train_df

    @property
    def test(self):
        return self._test_df
