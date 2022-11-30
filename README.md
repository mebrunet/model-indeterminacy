# model-indeterminacy
Source code accompanying the paper
[Implications of Model Indeterminacy for Explanations of Automated Decisions](https://openreview.net/forum?id=LzbrVf-l0Xq)

## Preamble
**Update 2022-11-29** - The code release is behind schedule.
Expect to see new files and edits over the coming few weeks.
I'll leave a note here when I get to an official release.

This repository is a pared-down version of the code used internally for the project.
We have done our best to ensure that it is complete and bug-free,
however it is very possible that I broke things as I trimmed it down for public release.
Don't hesitate to open a GitHub issue and I will attempt to assist you.

## Setup the codebase
Clone the repository, then run

```bash
conda env update -n indeterminacy -f environment_xxx.yml  # choose env.yml file for your system
# This creates a conda environment and install dependencies,
# note the paper results came from a linux machine with environment_x86_64.yml
conda activate indeterminacy  # activate the environment
conda develop src  # makes the source code importable
```

## Dataset prep
### Download the datasets
They are available on Kaggle.
Please read and follow all the applicable rules, terms, and conditions.
1. [UCI Credit Card](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
2. [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
3. [Porto Seguro's Safe Driver](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)

Unzip them and place the contents all in the same folder.
The contents of that folder should look like:
```
GiveMeSomeCredit/
  - Data Dictionary.xls
  - cs-test.csv
  - cs-training.csv
  - sampleEntry.csv
UCI_Credit_Card/
  - UCI_Credit_Card.csv
porto-seguro-safe-driver-prediction/
  - sample_submission.csv
  - test.csv
  - train.csv
```
You may need to create the `UCI_Credit_Card` folder,
as the unzipped contents might just be the csv file.

### Configure your compute environment
Edit the yaml file in `config/compute/local.yaml`.
You'll need to specify where this raw data folder can be found,
as well as where to put the processed datasets, results, etc.

The project makes use of [Hydra](hydra.cc) for configuration,
so if you're familiar with that you can actually set it up to run on different compute environments.

### Preprocess the data
Run the 3 preprocessing scripts in `src/indeterminacy/data/preprocess/`.
These can also be run interactively via code cells in an editor that supports them,
or as scripts with `./scripts/preprocess_data.sh`.
Exploratory data analysis reports will be output in the configured results directory.

### Test that it worked
There's a quick check to make sure the data is loading properly
```bash
pytest test/test_data.py
```

## Training Models
(Code and instructions to come.)

## Generating Explanations
(Code and instructions to come.)

## Analysis
(Code and instructions to come.)
