'''Preprocess the Give Me Some Credit Dataset

Link: https://www.kaggle.com/c/GiveMeSomeCredit/data

Variable descriptions in the provided Excel sheet

Note this is a competition, so the test set does not have targets
'''

# %% Imports
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from omegaconf import OmegaConf


# %% Load conifg
config = OmegaConf.load('config/compute/local.yaml')  # assumes working directory is project root

# %% Set Globals
RAW_DIR = config.raw_data
DATA_DIR = config.datapath
RESULTS_DIR = config.resultpath
OUT_FOLDER = 'give_me_credit'  # don't change this, folder name is reused throughout codebase


# %% -- Note the test set does not have targets, so only use training data
dataset_path = os.path.join(RAW_DIR, 'GiveMeSomeCredit', 'cs-training.csv')
df = pd.read_csv(dataset_path).drop(columns=["Unnamed: 0"])


# %% Renaming -- shorten
df.rename(columns={'SeriousDlqin2yrs': 'target',
                   'RevolvingUtilizationOfUnsecuredLines': 'revolving',
                   'DebtRatio': 'debt_ratio',
                   'MonthlyIncome': 'income',
                   'NumberOfDependents': 'num_dependents',
                   'NumberRealEstateLoansOrLines': 'num_re_loans',
                   'NumberOfOpenCreditLinesAndLoans': 'num_credit_lines',
                   'NumberOfTime30-59DaysPastDueNotWorse': 'num_30_late',
                   'NumberOfTime60-89DaysPastDueNotWorse': 'num_60_late',
                   'NumberOfTimes90DaysLate': 'num_90_late',
                   }, inplace=True)


# %% Generate EDA Report
os.makedirs(os.path.join(RESULTS_DIR), exist_ok=True)
profile = ProfileReport(df, title="Give Me Some Credit", html={'style': {'full_width': True}},
                        sort=None)
profile.to_file(os.path.join(RESULTS_DIR, 'EDA_give_me_credit.html'))


# %% Drop rows with missing data
df = df.dropna()
print('Keeping', len(df), 'datapoints')


# %% Dataset Properties
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
