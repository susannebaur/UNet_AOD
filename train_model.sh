#!/bin/bash

###### 

set -x 

SET_ID_MODULE="config"
MODEL_SCRIPT_MODULE="model"

#############################

cd /spiritx-home/sbaur/python_scripts/MRV-project/NN_model/UNet/
###### run python scripts
python train.py "$SET_ID_MODULE" "$MODEL_SCRIPT_MODULE"

echo "ALL DONE"