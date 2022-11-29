import os

from .interface import TabularData


class GiveMeCreditData(TabularData):
    def __init__(self, datapath):

        self._datapath = datapath
        train_df, test_df = self.load_dataset(os.path.join(datapath, 'give_me_credit'))

        self._train_df = train_df
        self._test_df = test_df
        self._target = 'target'

    @property
    def binaries(self):
        return []

    @property
    def categoricals(self):
        return []

    @property
    def ordinals(self):
        return ['age',
                'num_30_late',
                'num_60_late',
                'num_90_late',
                'num_dependents',
                'num_re_loans',
                'num_credit_lines']

    @property
    def continuous(self):
        return ['revolving',
                'income',
                'debt_ratio']

    @property
    def immutables(self):
        return ['age',
                'num_dependents']

    @property
    def target(self):
        return self._target

    @property
    def train(self):
        return self._train_df

    @property
    def test(self):
        return self._test_df
