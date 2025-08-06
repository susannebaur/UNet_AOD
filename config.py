import os
import numpy as np
import sys

################################     model settings     ##############################################
LOC = 'HAL' #'spirit' 'HAL' # define which cluster to run on
DIMENSIONS = 'time_lat' # 'time_lat' 'time_lat_lon'
T_RES = 'month' # 'day' 'month'
ID = f'{T_RES}_{DIMENSIONS}_IPSL_skip' # skip stands for skip connections in the model architecture

BATCH_SIZE = 256
NUM_EPOCHS = 250
KERNEL_SIZE = 3
PADDING = 1
MODEL_TYPE = "UNet" #"CNN"
TRAIN_START='1950'
TRAIN_END='1979'
VAL_START='1980'
VAL_END='1999'

#time shift in months

if T_RES == 'month':
    TIME_SHIFT_AMOUNT = {
    'rsut': 1,   # x months forward
    'rlut': 1,   # x months forward
    'ta': -5     # 60 months (5 years) backward
    }
elif T_RES == 'day':
    TIME_SHIFT_AMOUNT = {
    'rsut': 365,   # 12 months forward
    'rlut': 365,   # 12 months forward
    'ta': -1825     # 60 months (5 years) backward
    }
else:
    raise ValueError("check time shift settings")
    

MODELS = [
 #   ["CNRM-CERFACS","CNRM-CM6-1"],
  #  ["CNRM-CERFACS","CNRM-ESM2-1"],
    ["IPSL","IPSL-CM6A-LR"],
  #  ["MPI-M", "MPI-ESM1-2-LR"]
]

VARIABLES_BASE = [
            'ta',
            'rsut',
            'rlut',
         #   'clt',
            'rsdscs',
            'rsutcs',
        #    'co2mass'
            ]

VARIABLES_MAX = [
            'ta_m',
            'rsut_m'
            ]

VARIABLES_ALL = VARIABLES_BASE+VARIABLES_MAX

# number of realisations to use
rs = 31

if T_RES == 'month':
    REALISATIONS = range(1,rs) #21
elif T_RES == 'day' and 'IPSL' in MODELS[0][1]:
    REALISATIONS = range(1,12)
else:
    raise ValueError("Setting numbers of realisations failed")


# Test settings
TEST_EXPGROUP = 'GeoMIP' #'CMIP' 
TEST_EXP = 'G6sulfur' #'G6sulfur' 'historical'

if TEST_EXPGROUP == 'CMIP':
    TEST_SETTINGS = {
    'TEST_REALISATION': '1',
    'TEST_RANGE': ['1970', '1999'],
    'TEST_YEAR': ['1993', '1994'],
    'TEST_MODEL': ["IPSL","IPSL-CM6A-LR"]
    }
elif TEST_EXPGROUP == 'GeoMIP':
    TEST_SETTINGS = {
    'TEST_REALISATION': '1',
    'TEST_RANGE': ['2020', '2049'],
    'TEST_YEAR': ['2031', '2032'],
    #'BASE_RANGE': ['2015', '2029'],
    'BASE_RANGE': ['1850', '2014'],
    'TEST_MODEL': ["IPSL","IPSL-CM6A-LR"]
  #  'TEST_MODEL': ["CNRM-CERFACS","CNRM-ESM2-1"]
        
    }

#saving the model
if LOC == 'HAL':
    MODEL_SAVE_PATH = f'/home/sbaur/model_save/{MODEL_TYPE}/'
    MODEL_RECUP_PATH = f'/hal-home/sbaur/model_save/{MODEL_TYPE}/'
    DATA_INPUT_PATH = f'/hal-home/sbaur/postpro_data/{T_RES}/{DIMENSIONS}/'
    if len(MODELS) > 1:
        DATA_POSTPRO_PATH = f'/net/nfs/ssd0/ssd3/sbaur/MRV-project/postpro_data/{T_RES}/{DIMENSIONS}/'
    else:
        DATA_POSTPRO_PATH = f'/net/nfs/ssd0/ssd3/sbaur/MRV-project/postpro_data/{T_RES}/{DIMENSIONS}/{MODELS[0][1]}/'
else:
    MODEL_SAVE_PATH = f'/homedata/sbaur/MRV-project/NN_model/{MODEL_TYPE}/'

DATA_SAVE_PATH = f'/homedata/sbaur/MRV-project/NN_model/{MODEL_TYPE}/'

MODEL_SAVE_NAME = f'bs{BATCH_SIZE}_mems{str(len(list(REALISATIONS)))}_models{len(MODELS)}_{TRAIN_START}-{TRAIN_END}_{ID}.pth'