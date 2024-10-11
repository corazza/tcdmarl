#!/bin/bash
export PYTHONPATH="${PWD}" ; cd /home/user.name/repos/temporal-causal-dmarl ; /usr/bin/env /home/user.name/anaconda3/envs/tcdmarl/bin/python /home/user.name/repos/temporal-causal-dmarl/tcdmarl/main.py --collection 13_05_runs --all-experiments --num-trials 10 --step-unit-factor 500
