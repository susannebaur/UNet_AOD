#!/bin/bash

###### 


set -x 

SET_ID_MODULE="config"

#############################

cd /spiritx-home/sbaur/python_scripts/MRV-project/NN_model/UNet/
###### run python scripts
python preprocessing.py "$SET_ID_MODULE"

echo "ALL DONE"