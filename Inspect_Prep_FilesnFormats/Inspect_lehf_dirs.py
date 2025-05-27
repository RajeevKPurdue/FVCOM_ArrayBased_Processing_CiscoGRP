import os
from glob import glob
import netCDF4 as nc

folder = "/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/2017"
files = sorted(glob(os.path.join(folder, "*.nc")))

if not files:
    print("No .nc files found in", folder)
else:
    for f in files:
        try:
            size_mb = os.path.getsize(f) / 1e6
            with nc.Dataset(f, "r") as ds:
                print(f"File: {os.path.basename(f)} (Size: {size_mb:.1f} MB)")
                if "nv" in ds.variables:
                    nv_var = ds.variables["nv"]
                    print("  nv shape:", nv_var.shape)
                if "lat" in ds.variables and "lon" in ds.variables:
                    print("  lat shape:", ds.variables["lat"].shape,
                          "lon shape:", ds.variables["lon"].shape)
                if "zeta" in ds.variables:
                    print("  zeta shape:", ds.variables["zeta"].shape)
                if "Times" in ds.variables:
                    print("  Times shape:", ds.variables["Times"].shape)
        except Exception as e:
            print(f"Could not read {f}: {e}")
        print("-"*60)

#%%

import os
from glob import glob
import netCDF4 as nc
import xarray as xr

# Define your output directory path
OUTPUT_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2'

# List all aggregated netCDF files matching the pattern
output_files = sorted(glob(os.path.join(OUTPUT_PATH, "spatial_summary_*.nc")))

for f in output_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"File: {os.path.basename(f)} - Size: {size_mb:.2f} MB")

    # Option 1: Using netCDF4 to inspect dimensions and variable shapes
    ds_nc = nc.Dataset(f, "r")
    print("Using netCDF4:")
    print(" Dimensions:")
    for dim in ds_nc.dimensions:
        print(f"  {dim}: {len(ds_nc.dimensions[dim])}")
    print(" Variables:")
    for var in ds_nc.variables:
        print(f"  {var}: {ds_nc.variables[var].shape}")
    if 'time' in ds_nc.variables:
        time_vals = ds_nc.variables['time'][:]
        # Attempt to get the full range; this may need further decoding if stored as strings
        print(" Time variable values:", time_vals)
    ds_nc.close()

    # Option 2: Using xarray for a higher-level view
    ds_xr = xr.open_dataset(f)
    print("Using xarray:")
    print(ds_xr)
    if 'time' in ds_xr.coords:
        time_values = ds_xr['time'].values
        if time_values.size > 0:
            # Convert to string for display if necessary
            start_time = str(time_values[0])
            end_time = str(time_values[-1])
            print(f" Time range: {start_time} to {end_time}")
    ds_xr.close()

    print("-" * 60)

#%%


import os
import glob
import netCDF4 as nc
import pandas as pd

# Set the directory containing the input files for a given year (e.g., 2017)
input_dir = "/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/lefh_2018"

# List and sort all NetCDF files in the directory
all_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
all_files = [f for f in all_files if not os.path.basename(f).startswith("erie_000")]

# Select a subset of consecutive files (adjust the slice as needed)
subset_files = all_files[-2:] #[:5]  # e.g., first 5 files

for file in subset_files:
    print(f"\nProcessing file: {file}")
    # Open the file using netCDF4
    ds_in = nc.Dataset(file, mode="r")

    # Convert the Times variable (assumed to be a char array) to a list of strings
    times_raw = ds_in.variables["Times"][:]  # shape (nt, nchar)
    # Each element is an array of characters; join them into a single string per record
    times_str = ["".join(t.astype(str)).strip() for t in times_raw]

    # Convert string times to pandas datetime objects.
    # Adjust the format string if needed.
    times_dt = pd.to_datetime(times_str, format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce")

    # Create a DataFrame to group by day
    df = pd.DataFrame({"time": times_dt})
    df["date"] = df["time"].dt.date

    # Group by unique date and count the number of time steps
    group_counts = df.groupby("date").size()

    print("Time steps per day in the input file:")
    print(group_counts)

    ds_in.close()

