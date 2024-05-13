#!/bin/bash
export PYTHONPATH="${PWD}" ; cd /home/jan.corazza/repos/temporal-causal-dmarl ; /usr/bin/env /home/jan.corazza/anaconda3/envs/tcdmarl/bin/python /home/jan.corazza/repos/temporal-causal-dmarl/tcdmarl/main.py --collection 13_05_runs --tlcd --experiment routing --num-trials 30  --step-unit-factor 100

export PYTHONPATH="${PWD}" ; cd /home/jan.corazza/repos/temporal-causal-dmarl ; /usr/bin/env /home/jan.corazza/anaconda3/envs/tcdmarl/bin/python /home/jan.corazza/repos/temporal-causal-dmarl/tcdmarl/main.py --collection 13_05_runs --experiment routing --num-trials 30 --step-unit-factor 100
