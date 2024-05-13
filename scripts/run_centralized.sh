#!/bin/bash
# $1 refers to the first command line argument
if [ -z "$1" ]
then
  echo "No collection argument supplied"
  exit 1
fi

export PYTHONPATH="${PWD}" ; cd /home/jan.corazza/repos/temporal-causal-dmarl ; /usr/bin/env /home/jan.corazza/anaconda3/envs/tcdmarl/bin/python /home/jan.corazza/repos/temporal-causal-dmarl/tcdmarl/main.py --collection $1 --tlcd --experiment centralized_routing --num-trials 30  --step-unit-factor 700

export PYTHONPATH="${PWD}" ; cd /home/jan.corazza/repos/temporal-causal-dmarl ; /usr/bin/env /home/jan.corazza/anaconda3/envs/tcdmarl/bin/python /home/jan.corazza/repos/temporal-causal-dmarl/tcdmarl/main.py --collection $1 --experiment centralized_routing --num-trials 30 --step-unit-factor 700
