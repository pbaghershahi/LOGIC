#!/bin/bash

python main.py -cff ./config/amazon_products.yaml -dss 1000 -mnen 200 -m logic -ub \
-wno -lm "meta-llama/Meta-Llama-3.1-8B-Instruct" --seed 4978 \
-pgp ./pretrained/gcn/gcn.pth -pdp ./output/gcn/data.pth -ppp ./pretrained/gcn/projector.pth