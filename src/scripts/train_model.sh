#!/bin/bash

python src/train.py  --config src/model/configs/wae_mmd.yml \
                    --log-dir log_dir \
                    --model wae_mmd \
                    --exp-name "Testing" \
                    --run-name "Test run name" \
