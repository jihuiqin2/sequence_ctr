#!/usr/bin/env bash

model_id='dien_taobao'

config_model_yaml='../config/model_config/DIEN.yaml'

config_dataset_yaml= '../config/dataset_config/taobao.yaml'

python common_demo.py \
  --model_id $model_id \
  --config_model_yaml $config_model_yaml \
  --config_dataset_yaml $config_dataset_yaml
