# UNET_AOD
Python scipts for predicting stratospheric Aerosol Optical Depth (AOD) using GCM output and a 1D UNet deep learning architecture

## Contents
```
UNET_AOD/
├── config.py        # Set ID and run configurations
├── funcs.py         # Contains common functions
├── preprocessing.py # Load in and prepare datasets
├── prepro_data.sh   # Script that runs preprocessing.py (to run on HAL cluster with GPU)
├── model.py         # Contains ClimateDataset and UNet model 
├── train.py         # Script that runs the training and saves the model with the best fit
├── train_model.sh   # Script that runs train.py (to run on HAL cluster with GPU)
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── 
```

## Getting started
1. Clone the repository:
    ```bash
    git clone https://github.com/susannebaur/UNET_AOD.git
    cd UNET_AOD
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Adjust settings and paramters in the config.py file
4. Run the data processing script
    ```bash
    ./prepro_data.sh
    ```
5. Train the model
    ```bash
    ./train_model.sh
    ```
6. Test the model fit
