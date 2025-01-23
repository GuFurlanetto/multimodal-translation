#!/bin/bash

python src/train.py  --config src/model/configs/wae_mmd.yml \
                    --log-dir $1 \
                    --model wae_mmd \
                    --exp-name $2 \
                    --run-name $3 \
