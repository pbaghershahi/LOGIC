#!/bin/bash

python main.py -cff "./config/amazon_products.yaml" -m gspell \
-pdp ./output/gcn/data.pth \
-pgp ./pretrained/gcn/gcn.pth \
-ppp ./pretrained/gcn/projector.pth