#!/usr/bin/env bash

for dataset in credit_card give_me_credit safe_driver; do
  python -m indeterminacy.data.preprocess.${dataset}
done
