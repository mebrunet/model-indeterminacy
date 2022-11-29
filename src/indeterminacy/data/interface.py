"""Adapated from https://github.com/carla-recourse/CARLA"""

import os
import warnings
from abc import ABC, abstractmethod

import pandas as pd


class TabularData(ABC):
    """
    Abstract class to interface with tabular datasets.
    """
    def load_dataset(self, dataset_dir):
        ''' Helper to load the tabular datasets
        '''
        train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'), index_col=0)
        test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'), index_col=0)

        for df in (train_df, test_df):
            for col in df.columns:
                if col in self.binaries:
                    df[col] = df[col].astype('bool')

                elif col in self.categoricals:
                    df[col] = df[col].astype('category')

                elif col in self.ordinals:
                    df[col] = df[col].astype('int')

                elif col in self.continuous:
                    df[col] = df[col].astype('float')

                elif col == 'target':
                    pass

                else:
                    warnings.warn(f'variable {col} does not have classified type')

        return train_df, test_df    

    @property
    @abstractmethod
    def binaries(self):
        """
        Provides the column names of binary data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all binary columns
        """
        pass

    @property
    @abstractmethod
    def categoricals(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method,
        e.g., sex_female

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass

    @property
    @abstractmethod
    def ordinals(self):
        """
        Provides the column names of ordinal data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all ordinal columns
        """
        pass

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def train(self):
        """
        The raw training data (as Dataframe) without encoding or normalization

        Returns
        -------
        pd.DataFrame
            Tabular training data with raw information
        """
        pass

    @property
    @abstractmethod
    def test(self):
        """
        The raw test data (as Dataframe) without encoding or normalization

        Returns
        -------
        pd.DataFrame
            Tabular test data with raw information
        """
        pass

    @property
    def train_data(self):
        return self.train.drop(columns=[self.target])

    @property
    def test_data(self):
        return self.test.drop(columns=[self.target])

    @property
    def train_target(self):
        return self.train[self.target].to_numpy()

    @property
    def test_target(self):
        return self.test[self.target].to_numpy()

    @property
    def info(self):
        return {'instances': len(self),
                'train': len(self.train),
                'test': len(self.test),
                'features': len(self.train.columns) - 1,
                'binary': len(self.binaries),
                'categorical': len(self.categoricals),
                'ordinal': len(self.ordinals),
                'continuous': len(self.continuous),
                'immutable': len(self.immutables),
                'features_ohe': len(pd.get_dummies(self.train).columns) - 1}

    def print_info(self):
        for key, val in self.info.items():
            print(f'{key}: {val}')

    def __len__(self):
        """
        Implement length for convenience
        """
        return len(self.train) + len(self.test)
