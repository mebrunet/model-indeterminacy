'''Class to encode / decode the datasets'''

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
import pandas as pd
import numpy as np

from .interface import TabularData


class TabularEncoder:
    def __init__(self, tabular_data, robust=False):
        assert isinstance(tabular_data, TabularData)
        self._tabular_data = tabular_data
        self._columns_in = tabular_data.train_data.columns.to_numpy()  # keep a template of the df
        self._robust = robust

        # Initialize
        ordinal_trans = StandardScaler()
        continuous_trans = RobustScaler() if robust else StandardScaler()
        categorical_trans = OneHotEncoder(sparse=False)

        # Fit
        ordinal_trans.fit(tabular_data.train_data[tabular_data.ordinals])
        continuous_trans.fit(tabular_data.train_data[tabular_data.continuous])
        categorical_trans.fit(tabular_data.train_data[tabular_data.categoricals])

        # Store
        self._ordinal_trans = ordinal_trans
        self._continuous_trans = continuous_trans
        self._categorical_trans = categorical_trans

        self._ordinal_cols = tabular_data.ordinals
        self._continuous_cols = tabular_data.continuous
        self._categorical_cols = tabular_data.categoricals
        self._categorical_cols_out = categorical_trans.get_feature_names_out()

        self._columns_out = tabular_data.train_data.drop(
            columns=self._categorical_cols).columns.to_numpy()
        self._columns_out = np.concatenate((self._columns_out, self._categorical_cols_out))

        # Save dtypes
        self._dtypes = {}
        for col in self._columns_in:
            self._dtypes[col] = tabular_data.train_data[col].dtype

    def encode(self, df):
        assert (df.columns.to_numpy() == self._columns_in).all()
        encoded = df.copy()

        if len(self._ordinal_cols) > 0:
            encoded[self._ordinal_cols] = self._ordinal_trans.transform(df[self._ordinal_cols])

        if len(self._continuous_cols) > 0:
            encoded[self._continuous_cols] = self._continuous_trans.transform(
                df[self._continuous_cols])

        if len(self._categorical_cols) > 0:
            ohe_vars = self._categorical_trans.transform(df[self._categorical_cols])
            encoded.drop(columns=self._categorical_cols, inplace=True)
            encoded[self._categorical_cols_out] = ohe_vars

        return encoded

    def decode(self, x):
        if isinstance(x, pd.DataFrame):
            assert (x.columns.to_numpy() == self._columns_out).all()
            decoded = x.copy()

        elif isinstance(x, np.ndarray):
            assert x.shape[1] == len(self._columns_out)
            decoded = pd.DataFrame(columns=self._columns_out, data=x)

        else:
            raise TypeError('Expects to decode pandas.DataFrame or numpy.array')

        if len(self._ordinal_cols) > 0:
            decoded[self._ordinal_cols] = self._ordinal_trans.inverse_transform(
                decoded[self._ordinal_cols])

        if len(self._continuous_cols) > 0:
            decoded[self._continuous_cols] = self._continuous_trans.inverse_transform(
                decoded[self._continuous_cols])

        if len(self._categorical_cols) > 0:
            inv_ohe_vars = self._categorical_trans.inverse_transform(
                decoded[self._categorical_cols_out])
            decoded.drop(columns=self._categorical_cols_out, inplace=True)
            decoded[self._categorical_cols] = inv_ohe_vars

        decoded = decoded[self._columns_in]  # reset order of columns

        # reassign proper data types
        for col, dtype in self._dtypes.items():
            if col in self._ordinal_cols:
                decoded[col] = decoded[col].round(0).astype(dtype)
            else:
                decoded[col] = decoded[col].astype(dtype)

        return decoded
