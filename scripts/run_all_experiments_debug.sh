#!/bin/bash
# $1 refers to the first command line argument
if [ -z "$1" ]
then
  echo "No collection argument supplied"
  exit 1
fi

export PYTHONPATH="${PWD}" ; python3 tcdmarl/tools/train.py --collection $1 --config configs/debug.json --all-experiments
