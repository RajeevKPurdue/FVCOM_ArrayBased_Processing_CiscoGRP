import os
import re
from glob import glob
import xarray as xr
import pandas as pd
import numpy as np


# ---------------------------
# Custom Function: Weighted Daily Mean for Duplicate Time Entries
# ---------------------------
def custom_daily_mean(group):
    """
    For a grouped dataset (for one day), if there are two time entries,
    assume the first represents 23 hours and the second 1 hour, then compute
    a weighted average for each variable. Otherwise, simply take the mean.
    """
    if group["time"].size == 2:
        # Assume the first time step represents 23 hours and the second 1 hour.
        w = np.array([23, 1])
        # Compute the weighted mean for each variable along the time dimension.
        temp_wmean = (group["temperature"] * w[:, None, None]).sum(dim="time") / w.sum()
        do_wmean = (group["dissolved_oxygen"] * w[:, None, None]).sum(dim="time") / w.sum()
        depth_wmean = (group["depth"] * w[:, None, None]).sum(dim="time") / w.sum()
        # Return a new Dataset for the day containing the weighted mean values.
        return xr.Dataset({
            "temperature": temp_wmean,
            "dissolved_oxygen": do_wmean,
            "depth": depth_wmean
        })
    else:
        # If not exactly 2 time entries, simply return the mean over time.
        return group.mean(dim="time")


# ---------------------------
# Helper Function: Extract Year from Filename
# ---------------------------
def extract_year_from_filename(filename):
    """
    Look for a four-digit number immediately followed by '.nc' at the end
    of the filename and return it as the year.
    """
    match = re.search(r'(\d{4})\.nc$', filename)
    return match.group(1) if match else None


# ---------------------------
# Configuration and File Selection
# ---------------------------
# List of years (as strings) that you want to process.
select_years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

# Directory that contains the already processed (aggregated) files.
# For example, these files might be named like "spatial_summary_2017.nc".
OUTPUT_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2'

# Get a list of all files matching the pattern "spatial_summary_*.nc" in the OUTPUT_PATH.
processed_files = glob(os.path.join(OUTPUT_PATH, "spatial_summary_*.nc"))

# Filter the list to only include files whose year (extracted from the filename) is in select_years.
files_to_process = [
    f for f in processed_files
    if extract_year_from_filename(os.path.basename(f)) in select_years
]

print("Files to process:", files_to_process)

# For testing on a single file, uncomment the next line:
# files_to_process = [files_to_process[0]]

# ---------------------------
# Processing: Group by Day and Compute Daily Aggregates
# ---------------------------
for file in files_to_process:
    print(f"Processing file: {file}")

    # Open the processed NetCDF file as an xarray Dataset.
    ds = xr.open_dataset(file)

    # Ensure the 'time' coordinate is in proper datetime format.
    ds["time"] = pd.to_datetime(ds["time"].values)

    # Create a new coordinate 'date' by flooring the 'time' values to the day.
    # This effectively drops the hour/minute/second information.
    ds = ds.assign_coords(date=ds.time.dt.floor("D"))

    # Group the dataset by the new 'date' coordinate and apply the custom_daily_mean function.
    ds_daily = ds.groupby("date").apply(custom_daily_mean)

    # Optionally, rename the 'date' coordinate back to 'time' for consistency.
    ds_daily = ds_daily.rename({"date": "time"})

    # Extract the year from the filename (e.g., "2017") for output naming.
    year = extract_year_from_filename(os.path.basename(file))
    output_daily_file = os.path.join(OUTPUT_PATH, f"spatial_summary_daily_{year}.nc")

    # Save the daily aggregated dataset to a new NetCDF file.
    ds_daily.to_netcdf(output_daily_file)
    print(f"Written aggregated daily NetCDF output: {output_daily_file}")
