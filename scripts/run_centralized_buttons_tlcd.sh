#!/bin/bash
# $1 refers to the first command line argument
if [ -z "$1" ]
then
  echo "No collection argument supplied"
  exit 1
fi

# Run with TL-CD
export PYTHONPATH="${PWD}" ; cd /home/user.name/repos/temporal-causal-dmarl ; /usr/bin/env /home/user.name/anaconda3/envs/tcdmarl/bin/python /home/user.name/repos/temporal-causal-dmarl/tcdmarl/main.py --collection $1 --tlcd --experiment centralized_buttons --num-trials 10  --step-unit-factor 1000
