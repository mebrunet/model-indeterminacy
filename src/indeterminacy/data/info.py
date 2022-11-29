'''Print dataset information'''

import socket

from indeterminacy import data

# %load_ext autoreload
# %autoreload 2

# %% Localize
if socket.gethostname() == 'mebmbp':
    DATASET_DIR = '/Volumes/Transcend/datasets'
else:
    DATASET_DIR = '/scratch/gobi1/mebrunet/datasets'


# %% Credit card
credit_card_data = data.CreditCardData(DATASET_DIR)
credit_card_data.print_info()
print('E[y=1] =', credit_card_data.train.target.mean())
train, test, encoder, raw = data.load_dataset('credit_card', DATASET_DIR)


# %% Give me Credit
give_me_credit_data = data.GiveMeCreditData(DATASET_DIR)
give_me_credit_data.print_info()
print('E[y=1] =', give_me_credit_data.train.target.mean())


# %% Safe Driver
safe_driver_data = data.SafeDriverData(DATASET_DIR)
safe_driver_data.print_info()
print('E[y=1] =', safe_driver_data.train.target.mean())
