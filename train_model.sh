#!/bin/bash

###### 

set -x 

SET_ID_MODULE="set_ID_HAL_adv"
MODEL_SCRIPT_MODULE="model_script_HAL_1D_adv_skip"

#############################

cd /spiritx-home/sbaur/python_scripts/MRV-project/NN_model/UNet/
###### run python scripts
python train.py "$SET_ID_MODULE" "$MODEL_SCRIPT_MODULE"

echo "ALL DONE"