'''Preprocess the Give Me Some Credit Dataset

Link: https://www.kaggle.com/c/GiveMeSomeCredit/data

Variable descriptions in the provided Excel sheet

Note this is a competition, so the test set does not have targets
'''

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport


# %% Globals - 
RAW_DIR = '/Volumes/Transcend/raw_data/GiveMeSomeCredit'
DATA_DIR = '/Volumes/Transcend/datasets'
OUT_FOLDER = 'give_me_credit'


# %% -- Note the test set does not have targets
df = pd.read_csv(os.path.join(RAW_DIR, 'cs-training.csv')).drop(columns=["Unnamed: 0"])


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
profile = ProfileReport(df, title="Give Me Some Credit", html={'style': {'full_width': True}},
                        sort=None)
profile.to_file('results/EDA_give_me_credit.html')


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
