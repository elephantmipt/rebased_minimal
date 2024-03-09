#!/bin/bash

pip install -r requirements.txt
# pip install deepspeed
pip install triton==2.2.0
cd flash_linear_attention && pip install -e . && cd ..
pip install -e .


python3 generate.py
