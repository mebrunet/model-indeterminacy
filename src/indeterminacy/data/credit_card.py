import os

from .interface import TabularData


class CreditCardData(TabularData):
    def __init__(self, datapath):

        self._datapath = datapath
        train_df, test_df = self.load_dataset(os.path.join(datapath, 'credit_card'))

        self._train_df = train_df
        self._test_df = test_df
        self._target = 'target'

    @property
    def binaries(self):
        return ['sex']

    @property
    def categoricals(self):
        return ['education',
                'marriage']

    @property
    def ordinals(self):
        return ['age',
                'pay_1',
                'pay_2',
                'pay_3',
                'pay_4',
                'pay_5',
                'pay_6']

    @property
    def continuous(self):
        return ['limit_bal',
                'bill_amt1',
                'bill_amt2',
                'bill_amt3',
                'bill_amt4',
                'bill_amt5',
                'bill_amt6',
                'pay_amt1',
                'pay_amt2',
                'pay_amt3',
                'pay_amt4',
                'pay_amt5',
                'pay_amt6']

    @property
    def immutables(self):
        return ['age',
                'sex',
                'marriage']

    @property
    def target(self):
        return self._target

    @property
    def train(self):
        return self._train_df

    @property
    def test(self):
        return self._test_df
