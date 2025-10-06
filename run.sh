#!/bin/bash

python main.py -cff "./config/cora.yaml" -dss 1000 -mnen 5 -m logic -nt 5 -gne 5 \
-wno -lm "meta-llama/Meta-Llama-3.1-8B-Instruct" \
-pcrw 0.5 \
-pcew 0.5 \
-pmiw 0.5 \
-pwb \
-adp
# -pgp /workspace/LOGIC/pretrained/liar-meta-llama/Meta-Llama-3.1-8B-Instruct-2025-10-06-00-29-39/gcn.pth \
# -pdp ./output/gcn/data.pth \