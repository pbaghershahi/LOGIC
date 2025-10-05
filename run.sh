#!/bin/bash

python main.py -cff "./config/liar.yaml" -mnen 2 -m logic -nt 5 -gne 5 \
-wno -lm "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--seed 4978 \
-pcrw 0.5 \
-pcew 0.5 \
-pmiw 0.5 \
-pwb
# -pgp ./pretrained/gcn/gcn.pth \
# -pdp ./output/gcn/data.pth \