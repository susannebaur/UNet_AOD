#!/bin/bash

###### 


set -x 

SET_ID_MODULE="set_ID_HAL_adv"

#############################

cd /spiritx-home/sbaur/python_scripts/MRV-project/NN_model/UNet/
###### run python scripts
python preprocessing.py "$SET_ID_MODULE"

echo "ALL DONE"