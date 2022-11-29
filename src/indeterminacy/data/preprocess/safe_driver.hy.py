'''Preprocess the Porto Seguro's Safe Driver Prediction Dataset

Link: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

According to Kaggle competition website:

The variable names refer to the origin of variable, but there is no reason to concern about
it. "Ind" is related to individual or driver, "reg" is related to region, "car" is related to car
itself and "calc" is an calculated feature.

Note this is a competition, so the test set does not have targets
'''

# %% Imports
import os

import numpy as np
import pandas as pd
from pandas.api import types
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport


# %% Globals
RAW_DIR = '/Volumes/research/datasets/raw_data/porto-seguro-safe-driver-prediction'  # edit this
DATA_DIR = '/Volumes/research/datasets'  # This should be the same folder for all datasets

OUT_FOLDER = 'safe_driver'  # don't change this, folder name is reused throughout codebase


# %% -- Note the test set does not have targets
df = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'), na_values=[-1]).drop(columns=['id'])


# %% -- Drop the calculated features (as per winning strategy)
calc_features = [feat for feat in df.columns if feat.startswith('ps_calc')]
df = df.drop(columns=calc_features)


# %% Rename to remove redundant "ps_" prefix
df.rename(lambda x: x[3:] if x.startswith('ps_') else x, axis='columns', inplace=True)


# %% Generate EDA Report
profile = ProfileReport(df, title="Safe Driver", html={'style': {'full_width': True}},
                        sort=None)
profile.to_file(os.path.join(RAW_DIR, 'EDA_safe_driver.html'))


# %% Address missing values
for col in df.columns:
    na_count = pd.isna(df[col]).sum()
    percent_na = 100 * na_count / len(df)
    if na_count > 0:
        print(f'{col}: {na_count} ({percent_na:.3f}%)')


# %% Column operations
for col in df.columns:
    na_count = pd.isna(df[col]).sum()
    percent_na = 100 * na_count / len(df)

    if (na_count > 0):
        # If categorical, will just make missing a new category
        if col.endswith('_cat'):
            print(col, 'has missing')
            # fill then cast to int to kill decimal places in the category names
            df[col] = df[col].fillna(-1).astype('int')

        # If ordinal / numerical drop if more than 1%
        elif (percent_na > 1):
            print('droping', col)
            df = df.drop(columns=[col])

# %% Drop car_11_cat because it has an unreasonable number of categories
df.drop(columns=['car_11_cat'], inplace=True)


# %% Drop rows with missing data
df = df.dropna()
print('Keeping', len(df), 'datapoints')


# %% Check column names for readability
for i, col in enumerate(df.columns):
    print(i, col)

# %% Print out binary
for col in df.columns:
    if col.endswith('_bin'):
        print(col)

# %% Print out ordinals
for col in df.columns:
    if col.split('__')[0].endswith('_cat') or col.endswith('_bin'):
        pass

    elif types.is_integer_dtype(df[col]):
        print(col)

# %% Print out continuous
for col in df.columns:
    if col.split('__')[0].endswith('_cat') or col.endswith('_bin'):
        pass

    elif types.is_float_dtype(df[col]):
        print(col)


# %% Dataset properties
print('Num features:', len(df.columns) - 1)
print('Num instances:', len(df))


# %% Split test/train
df_train, df_test = train_test_split(df, random_state=0, test_size=0.3, stratify=df.target)


# %% Save data
os.makedirs(os.path.join(DATA_DIR, OUT_FOLDER), exist_ok=True)
df_train.to_csv(os.path.join(DATA_DIR, OUT_FOLDER, 'train.csv'))
df_test.to_csv(os.path.join(DATA_DIR, OUT_FOLDER, 'test.csv'))


# %% Quick test
df_loaded = pd.read_csv(os.path.join(DATA_DIR, OUT_FOLDER, 'train.csv'), index_col=0)
assert np.allclose(df_loaded.to_numpy(), df_train.to_numpy())
