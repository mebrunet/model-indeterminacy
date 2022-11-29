'''Preprocess the UCI Credit Card Dataset

Reference: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/discussion/34608

This research employed a binary variable,default payment (Yes = 1, No = 0),
as the response variable.

This study reviewed the literature and used the following 23 variables as explanatory variables:

X1: Amount of the given credit (NT dollar):
it includes both the individual consumer credit and his/her family (supplementary) credit.

X2: Gender (1 = male; 2 = female).

X3: Education (1 = graduate school; 2 = university; 3 = high school; 0, 4, 5, 6 = others).

X4: Marital status (1 = married; 2 = single; 3 = divorce; 0=others).

X5: Age (year).

X6 - X11: History of past payment.
We tracked the past monthly payment records (from April to September, 2005) as follows:
X6 = the repayment status in September, 2005;
X7 = the repayment status in August, 2005; . . .;
X11 = the repayment status in April, 2005.

The measurement scale for the repayment status is:
-2: No consumption;
-1: Paid in full;
0: The use of revolving credit;
1 = payment delay for one month;
2 = payment delay for two months; . . .;
8 = payment delay for eight months;
9 = payment delay for nine months and above.

X12-X17: Amount of bill statement (NT dollar).
X12 = amount of bill statement in September, 2005;
X13 = amount of bill statement in August, 2005; . . .;
X17 = amount of bill statement in April, 2005.

X18-X23: Amount of previous payment (NT dollar).
X18 = amount paid in September, 2005;
X19 = amount paid in August, 2005; . . .;
X23 = amount paid in April, 2005.

Y: client's behavior; Y=0 then not default, Y=1 then default
'''

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pandas_profiling import ProfileReport


# %% Globals
RAW_DIR = '/Volumes/Transcend/raw_data'  # unzipped contents
DATA_DIR = '/Volumes/Transcend/datasets'
OUT_FOLDER = 'credit_card'


# %%
df = pd.read_csv(os.path.join(RAW_DIR, 'UCI_Credit_Card.csv')).drop(columns=['ID'])


# %% Renaming
df.rename(columns={'default.payment.next.month': 'target',
                   'PAY_0': 'PAY_1'}, inplace=True)

# All to lower case too
df.rename(str.lower, axis='columns', inplace=True)


# %% In this datset sex is binary, encode it that way (0=Male, 1=Female)
df['sex'] = df['sex'] - 1


# %% Generate EDA Report
profile = ProfileReport(df, title="Credit Card", html={'style': {'full_width': True}},
                        sort=None)
profile.to_file('results/EDA_credit_card.html')


# %% No missing values
assert len(df) == len(df.dropna())


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
df_loaded.head()
