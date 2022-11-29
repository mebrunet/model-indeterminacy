# model-indeterminacy
Source code accompanying the paper
[Implications of Model Indeterminacy for Explanations of Automated Decisions](https://openreview.net/forum?id=LzbrVf-l0Xq)

## Preamble
*Update 2022-11-29 - I am a little behind schedule getting
this out the door. Expect to see new files and edits over the coming few weeks.
I'll leave a note here when I get to an official release.

This repository is a pared-down version of the code used internally for the project.
We have done our best to ensure that it is complete and bug-free,
however it is very possible that I broke things as I trimmed it down for public release.
Don't hesitate to open a GitHub issue and I will attempt to assist you.

## Setup the codebase
Clone the repository, then run

```bash
# create a conda environment and install dependencies, choose env.yml file for your system
# note the paper results came from a linux machine with environment_x86_64.yml
conda env update -n indeterminacy -f environment_xxx.yml
conda activate indeterminacy  # activate the environment
conda develop src  # makes the source code importable
```

## Download the datasets
They are available on Kaggle. Please read and follow all the applicable rules, terms, and conditions.
1. [UCI Credit Card](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
2. [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
3. [Porto Seguro's Safe Driver](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)

## Preprocess the data
Run the preprocessing scripts in `src/indeterminacy/data/preprocess/`
These can also be run interactively via code cells in an editor that supports them.
You will need to edit the global variables in the scripts to point to the input folder with the
raw data downloads, and the desired output folder.


## Getting Started
(Instructions to come.)
