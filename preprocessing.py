import sys
import importlib
import funcs


print("########################0001_dataprep_HAL.py########################")

# Global placeholder
set_ID = None
def main():
    global set_ID  # Tell Python we're assigning to the global variable

    if len(sys.argv) < 2:
        print("Usage: python main_script.py <module_name>")
        sys.exit(1)

    set_ID_setting = sys.argv[1]

    # Import modules
    set_ID = importlib.import_module(set_ID_setting)

    #print(f"Loaded set_ID module: {set_ID_setting}")
    print(f"ID inside main(): {set_ID.ID}")

if __name__ == "__main__":
    main()


# settings
ID = set_ID.ID
print(ID)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using {device}")



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
REALISATIONS = set_ID.REALISATIONS
VARIABLES_BASE = set_ID.VARIABLES_BASE
VARIABLES_MAX = set_ID.VARIABLES_MAX
VARIABLES_ALL = set_ID.VARIABLES_ALL
MODEL_SAVE_PATH = set_ID.MODEL_SAVE_PATH
MODEL_SAVE_NAME = set_ID.MODEL_SAVE_NAME
DATA_POSTPRO_PATH = set_ID.DATA_POSTPRO_PATH
T_RES = set_ID.T_RES
TIME_SHIFT_AMOUNT = set_ID.TIME_SHIFT_AMOUNT


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
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


## Loading in data

ds_base=dict()
ds_raw_hist = dict()

def data_input(exp, ds_r, var):
    print(var)
    if T_RES == 'month':
        folder='Amon'
    elif T_RES == 'day':
        if var=='rsut':
            folder='CFday'
        else:
            folder='day'
    else:
        raise ValueError('check T_RES setting')
        
    ds_train=dict()
    ds_val=dict()
    realisations=dict()
    
    for m in MODELS:
        print(m)
        train_set = dict()
        val_set = dict()
        rel_list = list()
        
        for r in REALISATIONS:
            try:
                ds_r[r] = xr.open_mfdataset(f'/bdd/CMIP6/CMIP/{m[0]}/{m[1]}/{exp}/r{r}i*/{folder}/{var}/*/latest/{var}_{folder}_*.nc',
                                                engine="netcdf4")

                # preprocessing time  
                ds_r[r] = ds_r[r].sel(time = slice('1850','1999'))

                if T_RES == 'month': #if 'month' in ID:
                    ds_r[r]['time'] = pd.date_range("1850-01-01", freq='m', periods=1800)
                else:
                    ds_r[r]['time'] = pd.date_range("1850-01-01", freq='d', periods=54786)

                    # if 'max' in ID:
                    #     if var == 'rsut' or var == 'ta':
                    #         ds_r[r] = ds_r[r].resample(time="M").max()
                    #     elif var == 'rlut':
                    #         ds_r[r] = ds_r[r].resample(time="M").min()

                if var == 'co2mass':
                    # Co2 data comes in 1D -> broadcast to 3D:
                    lat = np.linspace(-90,90,64)
                    lon = np.linspace(0,357.5,128)
                    
                    # Expand and broadcast co2mass
                    co2_da = ds_r[r]['co2mass'].expand_dims({
                        'lat': lat,
                        'lon': lon
                    }).transpose('time', 'lat', 'lon') 
   
                    ds_r[r] = xr.Dataset({"co2mass":co2_da})
                else:
                    # remove time_bounds variable
                    ds_r[r] = funcs.check_time_bounds(ds_r[r])
                    ds_r[r] = funcs.check_time_bounds_2(ds_r[r])
                    # make sure dimensions are lat, lon
                    ds_r[r] = funcs.lat_lon_rename(ds_r[r])
                    # remove seasonality 
                    ds_r[r] = funcs.remove_seasonality(ds_r[r],ds_r[r])
                    # choose stratospheric pressure level
                    if var == 'ta':
                        ds_r[r] = ds_r[r].sel(plev=1000)

                # include time shift
                if 'lag' in ID:
                    if var in TIME_SHIFT_AMOUNT:
                        shift = TIME_SHIFT_AMOUNT[var]
                        ds_r[r][var] = ds_r[r][var].shift(time=shift).chunk({'time': -1})
                        ds_r[r][var] = ds_r[r][var].interpolate_na(dim='time', method='linear', fill_value="extrapolate")


                # if 'smooth' in ID:
                #     interim = ds_r[r].rolling(lat=15, center=True).mean()
                #     interim = interim.fillna(ds_r[r])
                #     ds_r[r] = interim

                # Separate into train and val sets
                train_set[r] = ds_r[r].sel(time=slice(TRAIN_START, TRAIN_END))
                val_set[r] = ds_r[r].sel(time=slice(VAL_START, VAL_END))
                
                rel_list.append(r)
            
            except Exception as e: print(e) 

        realisations[m[1]] = rel_list
                
        ds_train[m[1]] = train_set
        ds_val[m[1]] = val_set
                
        print(rel_list)
        print(f'{var} data loaded')
    return ds_train, ds_val, realisations

for var in VARIABLES_BASE:
    ds_base[f'train_{var}'], ds_base[f'val_{var}'], list_realisations = data_input('historical', ds_raw_hist, var)




##### FIXING INCOMPLETE DATA ######
if T_RES == 'month' and 'ta' in VARIABLES_BASE and any('CNRM-CM6-1' in model for model in MODELS):
    #################### ta does not exist for r23 #################################
    ds_base[f'train_ta']['CNRM-CM6-1'][23] = ds_base[f'train_ta']['CNRM-CM6-1'][22]
    ds_base[f'val_ta']['CNRM-CM6-1'][23] = ds_base[f'val_ta']['CNRM-CM6-1'][22]
if T_RES == 'month' and 'rsutcs' in VARIABLES_BASE and any('CNRM-CM6-1' in model for model in MODELS):
    #################### ta does not exist for r23 #################################
    ds_base[f'train_rsutcs']['CNRM-CM6-1'][23] = ds_base[f'train_rsutcs']['CNRM-CM6-1'][22]
    ds_base[f'val_rsutcs']['CNRM-CM6-1'][23] = ds_base[f'val_rsutcs']['CNRM-CM6-1'][22]

if T_RES == 'day' and 'ta' in VARIABLES_BASE and any('IPSL-CM6A-LR' in model for model in MODELS):
    #################### ta does not exist for all years for r4 #################################
    ds_base[f'train_ta']['IPSL-CM6A-LR'][3] = ds_base[f'train_ta']['IPSL-CM6A-LR'][4]
    ds_base[f'val_ta']['IPSL-CM6A-LR'][3] = ds_base[f'val_ta']['IPSL-CM6A-LR'][4]

#################### co2mass does not exist for all members of CNRM-CM6-1 #################################
source_member_index = 9  # co2mass available in r10
missing_member_indices = list(range(10, 30))  # r11 to r30 (0-indexed 10 to 29)
if T_RES == 'month' and 'co2mass' in VARIABLES_ALL and any('CNRM-CM6-1' in model for model in MODELS):
    for member in missing_member_indices:
        ds_base['train_co2mass']['CNRM-CM6-1'][member] = ds_base['train_co2mass']['CNRM-CM6-1'][source_member_index]
        ds_base['val_co2mass']['CNRM-CM6-1'][member] = ds_base['val_co2mass']['CNRM-CM6-1'][source_member_index]


## If max over longitude is set, we need to duplicate the loaded variables and save them with _m suffix
for var in VARIABLES_MAX:
    ds_base[f'train_{var}'] = ds_base[f'train_{var[:-2]}']
    ds_base[f'val_{var}'] = ds_base[f'val_{var[:-2]}']


#### input variable data processing ####

ds_prepro = dict()

def data_prepro(ds_r, var):
    ds_ = dict()
    for m in MODELS:
        # concat the members in 1 dataset
        ds_[m[1]] = funcs.concat_members(list_realisations[m[1]], 'time', ds_r[m[1]])

        if m != 'CNRM-CM6-1' or m != 'CNRM-ESM2-1':
            vorlage = xr.open_mfdataset(f'/homedata/sbaur/MRV-project/GCM/volc_aod/aggregates/aod_volcan_strato_v3_185001-203012_T127.nc',
                                                engine="netcdf4")
            ds_[m[1]] = ds_[m[1]].interp(lat=vorlage.lat,
                                           lon=vorlage.lon,
                                           method='linear')
                
        if 'lon' not in ID:
            #### to avg along lon but keep the map (coordinate)
            # xx = ds_[m[1]].mean('lon')['rsut'].as_numpy()
            # yy = np.dstack(128*(xx, xx))
            # ds_[m[1]] = xr.Dataset(data_vars=dict(
            #             rsut=(['lat', 'lon', 'time'], yy.transpose(1,2,0))),
            #             coords={'lat': ds_[m[1]]['lat'],
            #                     'lon': ds_[m[1]]['lon'],
            #                     'time': ds_[m[1]]['time']})

            #### to avg along lon without keeping the coordinate
            if var in VARIABLES_MAX:
                if var == 'rsut_m' or var == 'ta_m':
                    ds_[m[1]] = ds_[m[1]].max('lon')
                elif var == 'rlut_m':
                    ds_[m[1]] = ds_[m[1]].min('lon')
                else:
                    print('problem with max / min variable')
            else:
                ds_[m[1]] = ds_[m[1]].mean('lon')

    if len(MODELS)>1:
        print(f'concatenating {str(len(MODELS))} models')
        out = xr.concat([ds_[MODELS[i][1]] for i in range(0, len(MODELS))], dim='time')
        return out
    else:
        return ds_[m[1]]
        
for i in ['train', 'val']:
    for var in VARIABLES_ALL:
        ds_prepro[f'{i}_{var}'] = data_prepro(ds_base[f'{i}_{var}'], var)

print(ds_prepro['train_rsut'])


#### AOD data loading and processing ####

# AOD data for the 1980-1999 period
# choose data from 1979 to not loose data during the interpolation 

AOD = xr.open_mfdataset(f'/homedata/sbaur/MRV-project/GCM/volc_aod/aggregates/aod_volcan_strato_v3_185001-203012_T127.nc',
                                                engine="netcdf4")

AOD = AOD.rename({"record":"time"})
AOD['time'] = pd.date_range("1850-01-01", freq='m', periods=2172)
# turn monthly data to daily
if T_RES == 'month':
    pass
elif T_RES == 'day':
    AOD_res = AOD.resample(time='1D').interpolate("linear") # -> monthly and daily look identical when plotting global means and almost identical when plotting time average with lat-lon map
    #fill in missing January 1850 dates
    jan_dates = pd.date_range('1850-01-01', '1850-01-30', freq='D')
    mar_dates = pd.date_range('1850-03-01', '1850-03-30', freq='D')
    
    mar_data = AOD_res.sel(time=mar_dates)
    jan_data = mar_data.copy()
    jan_data['time'] = jan_dates
    AOD = xr.concat([AOD_res, jan_data], dim='time')
    
    # Sort time to restore chronological order
    AOD = AOD.sortby('time')
else:
    raise ValueError(f"Unrecognized T_RES format: {T_RES}")

#### only if avg along lon without keeping the coordinate!!!!
if 'lon' not in ID:
    print('Warning: You are loosing the lon dimension!!!')
    AOD = AOD.mean('lon')


### Smooth AOD data
print('smoothing AOD')
AOD_s = AOD.rolling(lat=15, center=True).mean()
AOD_s = AOD_s.fillna(AOD)
AOD = AOD_s

AOD_set=dict()

# years of interest for training
AOD_set['train'] = AOD.sel(time=slice(TRAIN_START, TRAIN_END))
# years of interest for validation
AOD_set['val'] = AOD.sel(time=slice(VAL_START, VAL_END))

total_nr_realisations = sum(len(v) for v in list_realisations.values())


# concat AOD datasets together to match time length of input (depends on number of realisations per model)
for i in ['train', 'val']:
    ### works only if all models have the same number of members
    ds_prepro[f'{i}_aod'] = xr.concat(total_nr_realisations*[AOD_set[i]], dim='time')


### check whether output (AOD) and the first input variable have the same dimensions in the train and val datasets
def compare_shapes(array1: xr.DataArray, array2: xr.DataArray):
    """Compares the shapes of two xarray DataArrays and adjusts array2 if needed."""
    if array1.shape != array2.shape:
        print('shape mismatch')
        print("aod: " + array2.shape)
        print("rsut: " + array1.shape)
        array2 = array2.transpose(*array1.dims)
        print('shape after transposing')
        print("aod: " + array2.shape)
        print("rsut: " + array1.shape)
    return array2

for i in ['train', 'val']:
    ds_prepro[f'{i}_aod']['aod'] = compare_shapes(ds_prepro[f'{i}_{VARIABLES_ALL[0]}'][VARIABLES_ALL[0]], ds_prepro[f'{i}_aod']['aod'])




print('saving data')
#### save prepped data


for i in ['train', 'val']:
    for var in VARIABLES_ALL +['aod']:
        # delete existing
        pathlib.Path(f'{DATA_POSTPRO_PATH}{var}_models{len(MODELS)}_{TRAIN_START}_{TRAIN_END}_{i}.nc').unlink(missing_ok=True)
        # export
        ds_prepro[f'{i}_{var}'].to_netcdf(f'{DATA_POSTPRO_PATH}{var}_models{len(MODELS)}_{TRAIN_START}_{TRAIN_END}_{i}.nc')
        print(f'{DATA_POSTPRO_PATH}{var}_models{len(MODELS)}_{TRAIN_START}_{TRAIN_END}_{i}.nc')