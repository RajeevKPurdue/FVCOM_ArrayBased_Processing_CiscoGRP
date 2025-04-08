import os
from glob import glob
import netCDF4 as nc
import xarray as xr
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_nc_files(folder, pattern="*.nc"):
    """Get sorted list of NetCDF files in the specified folder."""
    return sorted(glob(os.path.join(folder, pattern)))

def process_nc_file(file_path):
    """Process a single NetCDF file and print its details."""
    try:
        size_mb = os.path.getsize(file_path) / 1e6
        with nc.Dataset(file_path, "r") as ds:
            logging.info(f"File: {os.path.basename(file_path)} (Size: {size_mb:.1f} MB)")
            for var_name in ["nv", "lat", "lon", "zeta", "Times"]:
                if var_name in ds.variables:
                    logging.info(f"  {var_name} shape: {ds.variables[var_name].shape}")
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")

def process_files_in_folder(folder):
    """Process all NetCDF files in a folder."""
    files = get_nc_files(folder)
    if not files:
        logging.info(f"No .nc files found in {folder}")
    else:
        for f in files:
            process_nc_file(f)
            logging.info("-" * 60)

def main():
    folder = "/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/2017"
    process_files_in_folder(folder)
    
    # Additional processing as needed

if __name__ == "__main__":
    main()

