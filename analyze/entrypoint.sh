#!/bin/bash


python script/pre-process.py csv_optimals &
python script/pre-process.py csv_wrong &
python script/check_correlation dataset_optimals.csv &
python script/check_correlation dataset_wrong.csv &

