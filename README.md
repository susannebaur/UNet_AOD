# UNET_AOD
Python scipts for predicting stratospheric Aerosol Optical Depth (AOD) using GCM output and a 1D UNet deep learning architecture

The UNet is trained on stratospheric AOD from volcanic eruptions and is tested on solar geoeningeering experiment G6sulfur

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
└── testing
    ├── test_model.py # Run the model with some test data
    ├── plot_predicted_putput.ipynb # Plot the fit of the test result
    ├── model_performance_G6sulfur_91-92.ipynb # visualize the performance metrics of a test run on G6sulfur data
    └── model_performance_volc-eruptions.ipynb # visualize the perforamnce metrics of a test run on volcanic eruptions
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
6. Test the model fit (no GPUs needed)
    Test the UNet model (test configurations are set in config.py)
    ```
    python test_model.py
    ```
    Jupyter notebook to visually check the fit and calculate performance metrics:
    plot_predicted_output.ipynb
    Jupyter notebook to visualize the performance metrics
    ```
    model_performance_G6sulfur_91-92.ipynb
    ```
    or
    ```
    model_performance_volc-eruptions.ipynb
    ```
