#!/bin/bash

python main.py -cff ./config/cora.yaml -m logic -ub \
-wno -lm "meta-llama/Meta-Llama-3.1-8B-Instruct" --seed 4978 \
-pwb -nt 1 \
# -pdp /workspace/LOGIC/data/data_cora.pt \
# -pgp /workspace/LOGIC/pretrained/best_model_Cora.pt \
# -pne 0
# -ppp ./pretrained/gcn/projector.pth \