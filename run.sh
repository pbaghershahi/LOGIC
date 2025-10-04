#!/bin/bash

python main.py -cff "./config/amazon_products.yaml" -dss 1000 -mnen 1 -m logic -ub -nt 5 \
-wno -lm "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--seed 4978 \
-pgp ./pretrained/gcn/gcn.pth \
-pdp ./output/gcn/data.pth \
-pcrw 0.25 \
-pcew 0.5 \
-pmiw 0.5 \
-pwb