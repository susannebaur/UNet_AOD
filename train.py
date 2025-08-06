
import funcs as funcs
import sys
import importlib

print("########################0010_train_UNet_HAL.py########################")

# Global placeholder
set_ID = None
model_script = None
def main():
    global set_ID, model_script  # Tell Python we're assigning to the global variable

    if len(sys.argv) < 3:
        print("Usage: python main_script.py <module_name>")
        sys.exit(1)

    set_ID_setting = sys.argv[1]
    model_script_setting = sys.argv[2]

    # Import modules
    set_ID = importlib.import_module(set_ID_setting)
    model_script = importlib.import_module(model_script_setting)

    #print(f"Loaded set_ID module: {set_ID_setting}")
    #print(f"Loaded model_script module: {model_script_setting}")
    print(f"ID inside main(): {set_ID.ID}")

if __name__ == "__main__":
    main()

ID = set_ID.ID
BATCH_SIZE = set_ID.BATCH_SIZE
NUM_EPOCHS = set_ID.NUM_EPOCHS
KERNEL_SIZE = set_ID.KERNEL_SIZE
PADDING = set_ID.PADDING
MODEL_TYPE = set_ID.MODEL_TYPE
TRAIN_START = set_ID.TRAIN_START
TRAIN_END = set_ID.TRAIN_END
VAL_START = set_ID.VAL_START
VAL_END = set_ID.VAL_END
MODELS = set_ID.MODELS
VARIABLES_BASE = set_ID.VARIABLES_BASE
VARIABLES_MAX = set_ID.VARIABLES_MAX
VARIABLES_ALL = set_ID.VARIABLES_ALL
MODEL_SAVE_PATH = set_ID.MODEL_SAVE_PATH
MODEL_SAVE_NAME = set_ID.MODEL_SAVE_NAME
#DATA_INPUT_PATH = set_ID.DATA_INPUT_PATH
DATA_POSTPRO_PATH = set_ID.DATA_POSTPRO_PATH
T_RES = set_ID.T_RES

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# Computational modules 
# %matplotlib inline
import sys
import xarray as xr
import glob
import os
import numpy as np
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import re
from array import array
import importlib
from pathlib import Path
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Dictionary to store activations
activations = {}

def activation_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy()
        mean = np.mean(activations[name])
        std = np.std(activations[name])
     #   print(f"Layer {name}: Mean={mean:.4f}, Std={std:.4f}")
    return hook


print(MODEL_SAVE_PATH)

TRAIN_SAMPLE=dict()
VAL_SAMPLE=dict()

for var in VARIABLES_ALL + ['aod']:
    print(var)
    print(f'{DATA_POSTPRO_PATH}{var}_models{len(MODELS)}_{TRAIN_START}_{TRAIN_END}_train.nc')
    TRAIN_SAMPLE[var] = xr.open_mfdataset(f'{DATA_POSTPRO_PATH}{var}_models{len(MODELS)}_{TRAIN_START}_{TRAIN_END}_train.nc',
                                                engine="netcdf4").load()
    VAL_SAMPLE[var] = xr.open_mfdataset(f'{DATA_POSTPRO_PATH}{var}_models{len(MODELS)}_{TRAIN_START}_{TRAIN_END}_val.nc',
                                            engine="netcdf4").load()


print("data loaded")

# Training the model
importlib.reload(model_script)

if __name__ == "__main__":

    train_data = []
    val_data = []
    scalers = {}

    for var in VARIABLES_ALL:
        if var.endswith('_m'):
            # monthly component (from input_m, ta_m, etc.)
            base_var = var
            var_name = var.replace('_m', '')
        else:
            base_var = var
            var_name = var

        train_data.append(TRAIN_SAMPLE[base_var][var_name].values)
        val_data.append(VAL_SAMPLE[base_var][var_name].values)


    # Include AOD last
    train_data.append(TRAIN_SAMPLE['aod']['aod'].values)
    val_data.append(VAL_SAMPLE['aod']['aod'].values)

    print('train dataset')
    
    train_dataset = model_script.ClimateDataset(
        VARIABLES_ALL,
        *train_data,
        model_save_path=MODEL_SAVE_PATH,
        model_save_name=MODEL_SAVE_NAME,
        fit_scalers=True
    )

    for var in VARIABLES_ALL:
        scaler_path = f'{MODEL_SAVE_PATH}{MODEL_SAVE_NAME[:-4]}/{var}_scaler.pkl'
        scalers[f'{var}_scaler'] = joblib.load(scaler_path)

    print('val dataset')
    val_dataset = model_script.ClimateDataset(
        VARIABLES_ALL,
        *val_data,
        model_save_path=MODEL_SAVE_PATH,
        model_save_name=MODEL_SAVE_NAME,
        fit_scalers=False,
        **scalers
    )


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = model_script.UNet(in_channels=len(VARIABLES_ALL))#(KERNEL_SIZE, PADDING, MAX_AOD)
    # # Attach hooks to layers to print out mean and std dev
    # model.enc1[0].register_forward_hook(activation_hook("enc1"))
    # model.enc2[0].register_forward_hook(activation_hook("enc2"))
    # model.enc3[0].register_forward_hook(activation_hook("enc3"))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # learning rate
    
    model_script.train_model(model, train_loader, val_loader, criterion, optimizer, MODEL_SAVE_PATH, MODEL_SAVE_NAME, num_epochs=NUM_EPOCHS)






