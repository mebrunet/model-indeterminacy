import pytest
import torch
from pandas.api import types
import numpy as np

from indeterminacy import data


DATASET_DIR = '/Volumes/research/datasets'


@pytest.fixture
def credit_card_data():
    return data.CreditCardData(DATASET_DIR)


@pytest.fixture
def give_me_credit_data():
    return data.GiveMeCreditData(DATASET_DIR)


@pytest.fixture
def safe_driver_data():
    return data.SafeDriverData(DATASET_DIR)


def test_column_types(credit_card_data, give_me_credit_data, safe_driver_data):
    for dataset in (credit_card_data, give_me_credit_data, safe_driver_data):
        for df in (dataset.train, dataset.test):
            for col in dataset.binaries:
                assert types.is_bool_dtype(df[col])

            for col in dataset.categoricals:
                assert types.is_categorical_dtype(df[col])

            for col in dataset.ordinals:
                assert types.is_integer_dtype(df[col])

            for col in dataset.continuous:
                assert types.is_float_dtype(df[col])


def test_encoder(credit_card_data, give_me_credit_data, safe_driver_data):
    for dataset in (credit_card_data, give_me_credit_data, safe_driver_data):
        encoder = data.TabularEncoder(dataset)
        x_df = dataset.train_data.iloc[0:1000]
        encoded = encoder.encode(x_df)
        decoded = encoder.decode(encoded)
        assert (x_df.columns == decoded.columns).all()
        assert (x_df.dtypes == decoded.dtypes).all()
        for col in x_df.columns:
            if col in dataset.continuous:
                assert np.allclose(x_df[col].to_numpy(), decoded[col].to_numpy())

            else:
                # mask = ~(x_df[col] == decoded[col])
                # if mask.sum() > 0:
                #     print(decoded[col][mask])
                #     print(x_df[col][mask])
                assert (x_df[col] == decoded[col]).all()


def test_loaders():
    for name in ('credit_card', 'give_me_credit', 'safe_driver'):
        train, test, encoder, raw = data.load_dataset(name, DATASET_DIR)
        assert isinstance(train[0][0], torch.Tensor)
        assert isinstance(test[0][0], torch.Tensor)
