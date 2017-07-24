#!/bin/sh

# feature engineering
cd ./data_munging/ && python gen_train.py

# model training
cd ../model/ && python gbdt_lr.py

# make submissions
python make_submissions.py
