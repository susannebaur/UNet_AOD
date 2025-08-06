import sys
import os
sys.path.append(os.path.abspath('..'))

import model as model_script
import config as set_ID


ID = set_ID.ID
BATCH_SIZE = set_ID.BATCH_SIZE
NUM_EPOCHS = set_ID.NUM_EPOCHS
KERNEL_SIZE = set_ID.KERNEL_SIZE
PADDING = set_ID.PADDING
MODEL_TYPE = set_ID.MODEL_TYPE
TRAIN_START = set_ID.TRAIN_START
TRAIN_END = set_ID.TRAIN_END
TEST_EXPGROUP = set_ID.TEST_EXPGROUP
TEST_EXP = set_ID.TEST_EXP
TEST_SETTINGS = set_ID.TEST_SETTINGS
T_RES = set_ID.T_RES
VARIABLES_BASE = set_ID.VARIABLES_BASE
VARIABLES_MAX = set_ID.VARIABLES_MAX
VARIABLES_ALL = set_ID.VARIABLES_ALL

IN_CHANNELS = len(VARIABLES_ALL)
TIME_SHIFT_AMOUNT = set_ID.TIME_SHIFT_AMOUNT

MODEL_SAVE_PATH = set_ID.MODEL_SAVE_PATH
MODEL_RECUP_PATH = set_ID.MODEL_RECUP_PATH
MODEL_SAVE_NAME = set_ID.MODEL_SAVE_NAME
DATA_SAVE_PATH = set_ID.DATA_SAVE_PATH

print(ID)
print(TEST_EXPGROUP)
print(TEST_EXP)
print(TEST_SETTINGS['TEST_MODEL'][0])


# Computational modules 
# %matplotlib inline
import xarray as xr
from glob import glob
import numpy as np
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import re
import importlib
import pathlib
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# own modules
import funcs


xr.set_options(display_style='html')
xr.set_options(keep_attrs = True)


###### LOAD TEST DATA
ds_base=dict()
ds_raw_hist = dict()

def data_input(exp, ds_r, var):
    print(var)
    if 'month' in ID:
        folder='Amon'
    else:
        if var=='rsut':
            folder='CFday'
        else:
            folder='day'
    print(f'/bdd/CMIP6/{TEST_EXPGROUP}/{TEST_SETTINGS["TEST_MODEL"][0]}/{TEST_SETTINGS["TEST_MODEL"][1]}/{exp}/r{TEST_SETTINGS["TEST_REALISATION"]}i*/{folder}/{var}/*/v*/{var}_{folder}_*.nc')
    try:
        ds_test = xr.open_mfdataset(f'/bdd/CMIP6/{TEST_EXPGROUP}/{TEST_SETTINGS["TEST_MODEL"][0]}/{TEST_SETTINGS["TEST_MODEL"][1]}/{exp}/r{TEST_SETTINGS["TEST_REALISATION"]}i*/{folder}/{var}/*/v*/{var}_{folder}_*.nc',
                                                engine="netcdf4")
    except:
        ds_test = xr.open_mfdataset(f'/bdd/CMIP6/{TEST_EXPGROUP}/{TEST_SETTINGS["TEST_MODEL"][0]}/{TEST_SETTINGS["TEST_MODEL"][1]}/{exp}/r{TEST_SETTINGS["TEST_REALISATION"]}i*/{folder}/{var}/*/latest/{var}_{folder}_*.nc',
                                                engine="netcdf4")
    ds_sea = xr.open_mfdataset(f'/bdd/CMIP6/CMIP/{TEST_SETTINGS["TEST_MODEL"][0]}/{TEST_SETTINGS["TEST_MODEL"][1]}/historical/r1i*/{folder}/{var}/*/latest/{var}_{folder}_*.nc',
                                                engine="netcdf4")

    ds_sea = ds_sea.sel(time = slice(TEST_SETTINGS['BASE_RANGE'][0],TEST_SETTINGS['BASE_RANGE'][-1]))
    ds_test = ds_test.sel(time = slice(TEST_SETTINGS['TEST_RANGE'][0],TEST_SETTINGS['TEST_RANGE'][-1]))


    if var == 'co2mass':
        # Co2 data comes in 1D -> broadcast to 3D:
        lat = np.linspace(-90,90,64)
        lon = np.linspace(0,357.5,128)
        
        # Expand and broadcast co2mass
        co2_da_test = ds_test['co2mass'].expand_dims({
            'lat': lat,
            'lon': lon
        }).transpose('time', 'lat', 'lon')
        co2_da_sea = ds_sea['co2mass'].expand_dims({
            'lat': lat,
            'lon': lon
        }).transpose('time', 'lat', 'lon') 
    
        ds_test = xr.Dataset({"co2mass":co2_da_test})
    else:
        # remove seasonality
        ds_test = funcs.remove_seasonality(ds_sea, ds_test)
    
    # preprocessing time  
    if T_RES == 'month':
        ds_test['time'] = pd.date_range(f'{TEST_SETTINGS["TEST_RANGE"][0]}-01-01', freq='ME', periods=360) 
    else:
        ds_test['time'] = pd.date_range(f'{TEST_SETTINGS["TEST_RANGE"][0]}-01-01', freq='d', periods=10957)


    
    # # remove time_bounds variable
    ds_test = funcs.check_time_bounds(ds_test)
    ds_test = funcs.check_time_bounds_2(ds_test)
    # make sure dimensions are lat, lon
    ds_test = funcs.lat_lon_rename(ds_test)

    # choose stratospheric pressure level
    if var == 'ta':
        ds_test = ds_test.sel(plev=1000)

    # include time shift
    if 'lag' in ID:
        if var in TIME_SHIFT_AMOUNT:
            shift = TIME_SHIFT_AMOUNT[var]
            ds_test = ds_test.shift(time=shift).chunk({'time': -1})
            ds_test = ds_test.interpolate_na(dim='time', method='linear', fill_value="extrapolate")

    # if 'smooth' in ID:
    #     print('smoothing')
    #     interim = ds_test.rolling(lat=10, center=True).mean()
    #     interim = interim.fillna(ds_test)
    #     ds_test = interim

    ds_test = ds_test.sel(time = slice(TEST_SETTINGS["TEST_YEAR"][0], TEST_SETTINGS["TEST_YEAR"][-1]))

    # put it on the right grid
    if TEST_SETTINGS["TEST_MODEL"] != 'CNRM-CM6-1' or TEST_SETTINGS["TEST_MODEL"] != 'CNRM-ESM2-1':
        vorlage = xr.open_mfdataset(f'/homedata/sbaur/MRV-project/GCM/volc_aod/aggregates/aod_volcan_strato_v3_185001-203012_T127.nc',
                                                engine="netcdf4")
        ds_test = ds_test.interp(lat=vorlage.lat,
                                lon=vorlage.lon,
                                method='linear')

    return ds_test

for var in VARIABLES_BASE:
    ds_base[var] = data_input(TEST_EXP, ds_raw_hist, var)
    print(f'{var} done')


for var in VARIABLES_MAX:
    ds_base[f'{var}'] = ds_base[f'{var[:-2]}']



ds_prepro=dict()

for var in VARIABLES_ALL:
    if 'lon' not in ID:
        print('avg along lon')
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
                    ds_prepro[var] = ds_base[var].max('lon')
            elif var == 'rlut_m':
                ds_prepro[var] = ds_base[var].min('lon')
            else:
                print('problem with max / min variable')
        else:
            ds_prepro[var] = ds_base[var].mean('lon')




######### AOD DATA
# AOD data for the 1980-1999 period
# choose data from 1979 to not loose data during the interpolation 

print(TEST_SETTINGS["TEST_MODEL"])
print(TEST_SETTINGS["TEST_MODEL"][1])

if TEST_EXPGROUP == 'CMIP':
    AOD = xr.open_mfdataset(f'/homedata/sbaur/MRV-project/GCM/volc_aod/aggregates/aod_volcan_strato_v3_185001-203012_T127.nc',
                                                    engine="netcdf4")
    AOD = AOD.rename({"record":"time"})
    AOD['time'] = pd.date_range("1850-01-01", freq='ME', periods=2172)

elif TEST_EXPGROUP == 'GeoMIP':
    if TEST_SETTINGS["TEST_MODEL"][1] == 'CNRM-ESM2-1':
        files = sorted(glob("/homedata/sbaur/G6_data/G6sulfur/AOD_CNRM-ESM-2-1/py_aod_geomip_strato*.nc"))  
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
        time = pd.date_range("2015-01", "2100-12", freq="MS")
        AOD = ds.assign_coords(time=time)
    elif TEST_SETTINGS["TEST_MODEL"][1] == 'IPSL-CM6A-LR':
        ds = xr.open_mfdataset("/bdd/CMIP6/GeoMIP/IPSL/IPSL-CM6A-LR/G6sulfur/r1i1p1f1/Emon/od550aerso/gr/files/d20200709/od550aerso_Emon_IPSL-CM6A-LR_G6sulfur_r1i1p1f1_gr_202001-210012.nc", engine='netcdf4')

        #interpolate to CNRM grid
        vorlage = xr.open_mfdataset(f'/homedata/sbaur/MRV-project/GCM/volc_aod/aggregates/aod_volcan_strato_v3_185001-203012_T127.nc', engine="netcdf4")
        AOD = ds.interp(lat=vorlage.lat,
                        lon=vorlage.lon,
                        method='linear')
        AOD['aod'] = AOD['od550aerso']

    else:
        raise KeyError("AOD file for TEST_MODEL not defined")
else:
    raise KeyError("TEST_EXPGROUP does not exist")


# years of interest for test
# years of interest for test
cut_year_start = str(int(TEST_SETTINGS["TEST_YEAR"][0])-1)
cut_year_end = str(int(TEST_SETTINGS["TEST_YEAR"][1])+1)
AOD = AOD.sel(time=slice(cut_year_start, cut_year_end))
# turn monthly data to daily
if T_RES == 'month':
    pass
elif T_RES == 'day':
    AOD = AOD.resample(time='1D').interpolate("linear") # -> monthly and daily look identical when plotting global means and almost identical when plotting time average with lat-lon map
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


# years of interest for test
ds_prepro['aod'] = AOD.sel(time=slice(TEST_SETTINGS["TEST_YEAR"][0], TEST_SETTINGS["TEST_YEAR"][-1]))

if 'lon' not in ID:
    print('avg along lon')
    #### to avg along lon without keeping the coordinate
    ds_prepro['aod'] = ds_prepro['aod'].mean('lon')


print('smoothing AOD')
interim = ds_prepro['aod'].rolling(lat=15, center=True).mean()
interim = interim.fillna(ds_prepro['aod'])
ds_prepro['aod'] = interim


### check whether AOD and SWR train and val have the same dimensions
def compare_shapes(array1: xr.DataArray, array2: xr.DataArray):
    """Compares the shapes of two xarray DataArrays and raises an error if they don't match."""
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: {array1.shape} != {array2.shape}")

compare_shapes(ds_prepro['rsut']['rsut'], ds_prepro['aod']['aod'])


#### LOAD ALL DATA ######
TEST_DATA=dict()

for var in VARIABLES_ALL:
    if '_m' in var:
        TEST_DATA[var] = ds_prepro[var][var[:-2]].load()
    else:
        TEST_DATA[var] = ds_prepro[var][var].load()
TEST_DATA['aod'] = ds_prepro['aod']['aod'].load()


###### LOAD MODEL ######

print(MODEL_SAVE_NAME)
model = model_script.UNet(in_channels=IN_CHANNELS)#(KERNEL_SIZE, PADDING, MAX_AOD)
model.load_state_dict(torch.load(f'{MODEL_RECUP_PATH}{MODEL_SAVE_NAME[:-4]}/{MODEL_SAVE_NAME}'))
model.eval()


### Get activations

activations = {}

# Hook function to capture activations
def activation_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

# Attach hooks to layers
model.enc1[0].register_forward_hook(activation_hook("enc1"))
model.enc2[0].register_forward_hook(activation_hook("enc2"))
model.enc3[0].register_forward_hook(activation_hook("enc3"))



##### RUN MODEL ######


out_dir = f"{MODEL_RECUP_PATH}{MODEL_SAVE_NAME[:-4]}"
input_vars = []

for name in VARIABLES_ALL:
    data = TEST_DATA[name].values
    scaler = joblib.load(os.path.join(out_dir, f"{name}_scaler.pkl"))
    scaled = scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
    input_vars.append(scaled)

aod_data = TEST_DATA['aod'].values
model_input = np.stack(input_vars, axis=1)  # [N, C, L]
model_input = torch.tensor(model_input, dtype=torch.float32)



model.eval()
with torch.no_grad():
    predicted_aod = model(model_input)


##### POSTPRO AND EXPORT ######
rsut_output = xr.DataArray(
    data=TEST_DATA['rsut'].values.squeeze(),
    dims=["time","lat"],
    coords=dict(
        lat=TEST_DATA['rsut'].lat,
        time=TEST_DATA['rsut'].time,
    )
).to_dataset(name='rsut')

aod_output = xr.DataArray(
    data=predicted_aod.squeeze(),
    dims=["time","lat"],
    coords=dict(
        lat=TEST_DATA['rsut'].lat,
        time=TEST_DATA['rsut'].time,
    )
).to_dataset(name='aod')

aod_target = xr.DataArray(
    data=aod_data.squeeze(),
    dims=["time","lat"],
    coords=dict(
        lat=TEST_DATA['rsut'].lat,
        time=TEST_DATA['rsut'].time,
    )
).to_dataset(name='aod')

# delete exisiting file if exists
pathlib.Path(f'{DATA_SAVE_PATH}{T_RES}/rsut_test_data_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc').unlink(missing_ok=True)
pathlib.Path(f'{DATA_SAVE_PATH}{MODEL_SAVE_NAME[:-4]}').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{DATA_SAVE_PATH}{MODEL_SAVE_NAME[:-4]}/predicted_aod_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc').unlink(missing_ok=True)
pathlib.Path(f'{DATA_SAVE_PATH}{T_RES}/aod_test_data_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc').unlink(missing_ok=True)

# export
rsut_output.to_netcdf(f'{DATA_SAVE_PATH}{T_RES}/rsut_test_data_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc')
aod_output.to_netcdf(f'{DATA_SAVE_PATH}{MODEL_SAVE_NAME[:-4]}/predicted_aod_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc')
aod_target.to_netcdf(f'{DATA_SAVE_PATH}{T_RES}/aod_test_data_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc')

print(f'{DATA_SAVE_PATH}{MODEL_SAVE_NAME[:-4]}/predicted_aod_{TEST_SETTINGS['TEST_MODEL'][0]}_{TEST_SETTINGS["TEST_YEAR"][0]}-{TEST_SETTINGS["TEST_YEAR"][-1]}.nc')

# # Plot activations
# for layer_name, activation in activations.items():
#     plt.figure(figsize=(6, 4))
#     sns.histplot(activation.flatten(), bins=50, kde=True)
#     plt.title(f"Activation Distribution - {layer_name}")
#     plt.xlabel("Activation Value")
#     plt.ylabel("Frequency")
