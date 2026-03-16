#!/usr/bin/env bash

# Requires a trained checkpoint first.
python MVA/Test_on_toy_datasets/gmm_2d_flower.py \
  --config MVA/Test_on_toy_datasets/configs/gmm_2d_default.yaml
