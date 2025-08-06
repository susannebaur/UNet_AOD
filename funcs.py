import xarray as xr
import os
import pandas as pd


def check_time_bounds(data:xr.Dataset):
    """
    Removes the 'time_bnds' variable from an xarray Dataset if it exists.

    Parameters:
        data (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The Dataset with 'time_bounds' dropped if present; otherwise, returns the original Dataset.
    """
    return data.drop('time_bnds') if 'time_bnds' in data.data_vars else data


def check_time_bounds_2(data:xr.Dataset):
    """
    Removes the 'time_bounds' variable from an xarray Dataset if it exists.

    Parameters:
        data (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The Dataset with 'time_bounds' dropped if present; otherwise, returns the original Dataset.
    """
    return data.drop('time_bounds') if 'time_bounds' in data.data_vars else data


def lat_lon_rename(data:xr.Dataset):
    """
    Renames the 'longitude' and 'latitude' coordinates in an xarray Dataset to 'lon' and 'lat', respectively.

    Parameters
    ----------
    data : xr.Dataset
        The input xarray Dataset containing 'longitude' and 'latitude' coordinates.

    Returns
    -------
    xr.Dataset
        The Dataset with coordinates renamed to 'lon' and 'lat'. If the coordinates do not exist, returns the original Dataset unchanged.
    """
    try:
        data = data.rename({'longitude':'lon', 'latitude': 'lat'})
    except:
        pass
    return data


def concat_members(list_realisations, concat_dim, data:xr.Dataset):
    """
    Concatenates specified members from an xarray Dataset along a given dimension.

    Parameters:
        list_realisations (list of str): List of realisation numbers to concatenate from the Dataset (e.g., [0,1,2,3]).
        concat_dim (str): Name of the dimension along which to concatenate the members.
        data (xr.Dataset): The xarray Dataset containing the members to be concatenated.

    Returns:
        xr.DataArray: Concatenated DataArray of the selected members along the specified dimension.
    """
    lr = list_realisations
    return xr.concat([data[lr[i]] for i in range(0, len(lr))], dim=concat_dim)


def remove_seasonality(data_base:xr.Dataset, data:xr.Dataset):
    """
    Removes the mean seasonal cycle from the input data using a reference dataset.

    Parameters
    ----------
    data_base : xr.Dataset
        Reference dataset used to compute the mean seasonal cycle. Must contain a 'time' coordinate with a 'month' attribute. Returns a map for each month.
    data : xr.Dataset
        Dataset from which seasonality will be removed. Must contain a 'time' coordinate with a 'month' attribute.

    Returns
    -------
    xr.Dataset
        The input data with the mean seasonal cycle (computed from `data_base`) removed.
    The result is a Dataset where each month's data is adjusted by subtracting the mean value for that month from the reference dataset.
    """
    seasonality = data_base.groupby("time.month").mean("time")
    return (data.groupby("time.month") - seasonality).drop('month')


def remove_longitude(data:xr.Dataset, cfg):
    """
    Removes the longitude dimension from an xarray Dataset by either taking the maximum or mean along the 'lon' axis.

    Parameters:
        data (xr.Dataset): The input xarray Dataset containing a 'lon' dimension.
        cfg (dict): Configuration dictionary. If 'max' key is True, the maximum value along 'lon' is returned;
                    otherwise, the mean value along 'lon' is returned.

    Returns:
        xr.Dataset: The Dataset with the 'lon' dimension reduced by either maximum or mean.
    """
    if cfg.get("max", False):
        return data.max('lon')
    else:
        return data.mean('lon')


def average_metrics_6month(metric_dict):
    from collections import defaultdict
    """
    Groups monthly metric data into 6-month periods and computes the average for each.
    
    Args:
        metric_dict (dict): Dictionary with keys as 'YYYY-MM' and values as float.
    
    Returns:
        dict: Dictionary with 6-month labels as keys and average values as values.
    """
    grouped = defaultdict(list)

    for date_str, value in metric_dict.items():
        year, month = date_str.split('-')
        month = int(month)
        
        # Determine 6-month period
        if 1 <= month <= 6:
            period = f'{year}-H1'  # Jan to June
        else:
            period = f'{year}-H2'  # July to Dec
        
        grouped[period].append(value)

    # Average values
    avg_metrics = {period: round(sum(vals) / len(vals), 2) for period, vals in grouped.items()}
    return avg_metrics


def log_metrics(model_name, run_id, correlation, rmse, rmse_pct, csv_file):
    from datetime import datetime
    import csv
    """
    Logs the model performance metrics to a CSV file.
    
    Parameters:
    model_name (str): Name or identifier of the model.
    run_id (str): Unique identifier for the run (e.g., set in set_ID).
    correlation (float): Pearson correlation coefficient.
    rmse (float): Root Mean Squared Error.
    csv_file (str): Path to the CSV file where metrics will be logged.
    """
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)
    print(csv_file)

    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        if run_id in df_existing['ID'].values:
            print(f"⚠️ Warning: Model ID '{run_id}' already exists in the log file!")
    
    # Define the header and the data row
    header = ['Timestamp', 'Model_Name', 'ID', 'Pearson_Correlation_6moAvg', 'RMSE_6moAvg', 'RMSE_6moAvg_pct']
    data_row = [
        datetime.now().date().isoformat(),
        model_name,
        run_id,
        str(correlation),
        str(rmse),
        str(rmse_pct)
    ]
    
    # Open the CSV file in append mode and write the data
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file is new
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)