'''Classes to load the datasets in Pytorch fashion'''

# import torch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import CreditCardData, GiveMeCreditData, SafeDriverData, TabularEncoder


class DataFrameDataset(Dataset):
    def __init__(self, df, target, classes=None, encoder=None):
        assert isinstance(df, pd.DataFrame)
        if isinstance(target, str):
            # Tear off target y
            self.targets = torch.tensor(df[target].to_numpy())
            X = df.drop(columns=[target])

        else:
            assert isinstance(target, np.ndarray)
            assert len(target) == len(df)
            self.targets = torch.tensor(target)
            X = df

        if classes is None:
            classes = list(self.targets.unique())
            self.classes = classes

        if encoder is not None:
            X = encoder.encode(X)

        # print(X.dtypes)
        self._feature_names = X.columns
        X = torch.tensor(X.to_numpy(dtype='float'), dtype=torch.float32)
        self.data = X
        self._encoder = encoder  # Convenience

    def decoded(self, idx):
        x = self.data[idx].unsqueeze(0)
        df = self._encoder.decode(x.cpu().numpy())
        df['target'] = self.targets[idx].item()
        return df

    @property
    def feature_names(self):
        return self._feature_names.to_list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        return X, y


def load_dataset(name, datadir, robust_encoding=False):
    if name == 'credit_card':
        dataset = CreditCardData(datadir)

    elif name == 'give_me_credit':
        dataset = GiveMeCreditData(datadir)

    elif name == 'safe_driver':
        dataset = SafeDriverData(datadir)

    else:
        raise NotImplementedError(f'{name} not an implemented dataset')

    encoder = TabularEncoder(dataset, robust_encoding)

    train = DataFrameDataset(dataset.train_data, dataset.train_target, encoder=encoder)
    test = DataFrameDataset(dataset.test_data, dataset.test_target, encoder=encoder)

    return train, test, encoder, dataset
