"""
Created on Mon Jan 27 15:07:34 2025


Converting R code to process FVCOM years due to python memory efficiency +
    altering for volumetric weighting

Var and File info: Unknown start date meta data format
    Surface_PrecipEvap_Forcing: SURFACE PRECIPITATION FORCING IS OFF
    dimensions(sizes): nele(11509), node(6106), siglay(20), siglev(21), three(3), time(145 - Varies with files), DateStrLen(26)
    variables(dimensions): float32 x(node), float32 y(node), float32 lon(node),
     float32 lat(node), float32 siglay(siglay, node),
     float32 siglev(siglev, node), float32 h(node), int32 nv(three, nele),
     float32 zeta(time, node), float32 time(time),
     float32 temp(time, siglay, node),
     float32 Dissolved_oxygen(time, siglay, node),

@author: rajeevkumar
"""
import numpy as np
import netCDF4 as nc
import dask.array as da
import os
import re
import logging
import pandas as pd
from dask.distributed import Client, LocalCluster
from glob import glob
import xarray as xr
import pyproj

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share'
OUTPUT_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2'
TRIANGLE_AREAS_PATH = os.path.join(OUTPUT_PATH, 'triangle_areas.npy')
GRID_FILE = os.path.join(OUTPUT_PATH, 'grid.nc')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Option: set to True to force overwriting the grid file (e.g., if node attributes change)
OVERWRITE_GRID = False

# Set up Dask cluster for parallelization
def setup_dask(port):
    cluster = LocalCluster(
        n_workers=6,
        threads_per_worker=1,
        memory_limit="8GB",
        local_directory="/Volumes/WD Backup/dask_tmp",
        dashboard_address=f":{port}"
    )
    return Client(cluster)


# Extract year from folder name
def extract_year(folder_name):
    match = re.search(r'_(\d{4})$', folder_name)
    return match.group(1) if match else folder_name


# Compute triangle areas using Pyproj to transform node coordinates and Heron's formula
def compute_triangle_areas(lon, lat, nv):
    logger.info("Computing triangle areas using projected coordinates...")
    # Create a transformer: from EPSG:4326 (lon/lat) to EPSG:32617 (UTM zone 17N, in meters)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    # Transform all node coordinates from degrees to meters
    x, y = transformer.transform(lon, lat)
    # Use the connectivity array (nv) to get the coordinates of each triangle's vertices
    p1 = np.stack((x[nv[:, 0]], y[nv[:, 0]]), axis=1)
    p2 = np.stack((x[nv[:, 1]], y[nv[:, 1]]), axis=1)
    p3 = np.stack((x[nv[:, 2]], y[nv[:, 2]]), axis=1)
    # Compute edge lengths in meters using Euclidean norm
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p3, axis=1)
    c = np.linalg.norm(p3 - p1, axis=1)
    s = (a + b + c) / 2
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    print("Computed triangle_areas shape:", areas.shape)
    np.save(TRIANGLE_AREAS_PATH, areas)
    return areas


# Write grid file (including triangle areas, node coordinates, connectivity, and bathymetry)
def write_grid_file(lon, lat, nv, h, triangle_areas):
    logger.info("Writing grid file...")
    ds_grid = nc.Dataset(GRID_FILE, "w", format="NETCDF4")
    ds_grid.createDimension("nele", nv.shape[0])
    ds_grid.createDimension("three", nv.shape[1])
    ds_grid.createDimension("node", lon.shape[0])

    area_var = ds_grid.createVariable("triangle_areas", "f4", ("nele",))
    nv_var = ds_grid.createVariable("nv", "i4", ("nele", "three"))
    lon_var = ds_grid.createVariable("lon", "f4", ("node",))
    lat_var = ds_grid.createVariable("lat", "f4", ("node",))
    h_var = ds_grid.createVariable("h", "f4", ("node",))

    area_var[:] = triangle_areas
    nv_var[:] = nv  # nv should already be transposed and zero-indexed
    lon_var[:] = lon
    lat_var[:] = lat
    h_var[:] = h

    ds_grid.close()


# Process one year's FVCOM NetCDF files and aggregate daily cell-based data
def process_fvcom_year(year_path, year_label, client, lon, lat, nv, h, siglay, triangle_areas):
    logger.info(f"Processing Year: {year_label} at {year_path}")
    file_list = sorted(glob(os.path.join(year_path, "*.nc")))
    #######---------------------######## FOR LEFH SHARE WITH DIFF FILE NAMES
    file_list = [f for f in file_list if not os.path.basename(f).startswith("erie_000")]
    #######---------------------########
    if not file_list:
        logger.warning(f"No files found in {year_path}. Skipping.")
        return

    try:
        ds = nc.Dataset(file_list[0], mode="r")
        # Read grid and time variables from the file
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:] - 360
        Times = ds.variables['Times'][:, :]
        time = np.array(["".join(char.decode("utf-8") for char in row).strip() for row in Times])
        time = pd.to_datetime(time, format="%Y-%m-%dT%H:%M:%S.%f", utc=True, errors='coerce')
        mask_valid = ~pd.isnull(time)
        time = time[mask_valid]
        Times = Times[mask_valid]
        daychar = time.strftime("%Y-%m-%d")
        days = np.unique(daychar)

        # Read dynamic variables from the file
        temp = ds.variables['temp'][:]  # expected shape: (time, siglay, node)
        do = ds.variables['Dissolved_oxygen'][:]  # expected shape: (time, siglay, node)
        zeta = ds.variables['zeta'][:]  # expected shape: (time, node)
        h = ds.variables['h'][:]  # shape: (node,)
        siglay = ds.variables['siglay'][:, 0]  # convert to 1D array; shape: (siglay,)

        logger.info(f"h shape: {h.shape}, zeta shape: {zeta.shape}, siglay shape: {siglay.shape}")

        # Initialize lists to collect aggregated daily data
        all_temp = []  # will have shape (num_days, nele, num_layers)
        all_do = []  # will have shape (num_days, nele, num_layers)
        all_depth = []  # will have shape (num_days, nele, num_layers)
        all_dates = []  # list of date strings
        num_layers = siglay.shape[0]

        for day in days:
            indices = np.where(daychar == day)[0]

            # Calculate daily means and transpose so that the node dimension comes first
            temp_day = np.mean(temp[indices, :, :], axis=0).T  # shape becomes (node, siglay)
            do_day = np.mean(do[indices, :, :], axis=0).T  # shape becomes (node, siglay)
            zeta_day = np.mean(zeta[indices, :], axis=0)  # shape: (node,)

            # Compute actual depth at each node and layer (depth = h * siglay + zeta_day)
            depth = h[:, np.newaxis] * siglay[np.newaxis, :] + zeta_day[:, np.newaxis]  # shape: (node, siglay)

            # Compute cell-based averages over the triangles for each layer
            cell_temp_3d = np.zeros((nv.shape[0], num_layers))
            cell_do_3d = np.zeros((nv.shape[0], num_layers))
            cell_depth_3d = np.zeros((nv.shape[0], num_layers))

            for i in range(num_layers):
                layer_depth = depth[:, i]  # (node,)
                # Average the depth over the three nodes per cell
                mean_depth = np.mean(layer_depth[nv], axis=1)
                mean_temp = np.mean(temp_day[nv, i], axis=1)
                mean_do = np.mean(do_day[nv, i], axis=1)

                cell_temp_3d[:, i] = mean_temp
                cell_do_3d[:, i] = mean_do
                cell_depth_3d[:, i] = mean_depth

            # Append daily results to lists
            all_temp.append(cell_temp_3d)
            all_do.append(cell_do_3d)
            all_depth.append(cell_depth_3d)
            all_dates.append(day)

        # Convert lists to numpy arrays with a new time axis:
        all_temp = np.stack(all_temp, axis=0)  # shape: (num_days, nele, num_layers)
        all_do = np.stack(all_do, axis=0)  # shape: (num_days, nele, num_layers)
        all_depth = np.stack(all_depth, axis=0)  # shape: (num_days, nele, num_layers)

        # Create an xarray Dataset for final output with dimensions (time, nele, num_layers)
        ds_out = xr.Dataset(
            {
                "temperature": (("time", "nele", "num_layers"), all_temp),
                "dissolved_oxygen": (("time", "nele", "num_layers"), all_do),
                "depth": (("time", "nele", "num_layers"), all_depth)
            },
            coords={
                "time": all_dates,
                "nele": np.arange(all_temp.shape[1]),
                "num_layers": np.arange(all_temp.shape[2])
            }
        )

        # Write the aggregated dataset as a netCDF file
        output_nc_file = os.path.join(OUTPUT_PATH, f"spatial_summary_{year_label}.nc")
        ds_out.to_netcdf(output_nc_file)
        logger.info(f"Written aggregated netCDF output: {output_nc_file}")

    except Exception as e:
        logger.error(f"Error processing {year_label}: {e}", exc_info=True)
    finally:
        ds.close()



# Main execution block - LEFH + TRYING NOT TO OVERWRITE GRID FILES
if __name__ == "__main__":
    year_dirs = sorted([
        os.path.join(EXTERNAL_DRIVE_PATH, d)
        for d in os.listdir(EXTERNAL_DRIVE_PATH)
        if os.path.isdir(os.path.join(EXTERNAL_DRIVE_PATH, d))
    ])

    base_port = 8787

    # Use grid information from the first file of the first year directory that meets our filter
    first_year_dir = year_dirs[0]
    first_file_list = sorted(glob(os.path.join(first_year_dir, "*.nc")))
    first_file_list = [f for f in first_file_list if not os.path.basename(f).startswith("erie_000")]
    first_ds = nc.Dataset(first_file_list[0], mode="r")
    lon = first_ds.variables['lon'][:] - 360
    lat = first_ds.variables['lat'][:]
    nv = first_ds.variables['nv'][:].T - 1  # Convert from (3, nele) to (nele, 3)
    h = first_ds.variables['h'][:]
    # For sigma layers, extract a representative column (assumes sigma values are the same at all nodes)
    siglay = first_ds.variables['siglay'][:, 0]
    first_ds.close()

    # Compute or load triangle areas using the new projected approach
    if os.path.exists(TRIANGLE_AREAS_PATH):
        triangle_areas = np.load(TRIANGLE_AREAS_PATH)
    else:
        triangle_areas = compute_triangle_areas(lon, lat, nv)

    # Write the grid file if it does not exist or if we want to overwrite it
    if OVERWRITE_GRID or not os.path.exists(GRID_FILE):
        write_grid_file(lon, lat, nv, h, triangle_areas)
        logger.info(f"Grid file written to {GRID_FILE}")
    else:
        logger.info(f"Grid file already exists at {GRID_FILE}. Skipping grid file write.")

    # Process each year's data
    for i, year_dir in enumerate(year_dirs):
        year_label = extract_year(os.path.basename(year_dir))
        client = setup_dask(base_port + i)
        process_fvcom_year(year_dir, year_label, client, lon, lat, nv, h, siglay, triangle_areas)
        client.close()




""""
# Main execution block - SUCCESSFUL RUN ON OG DIRS
if __name__ == "__main__":
    year_dirs = sorted([
        os.path.join(EXTERNAL_DRIVE_PATH, d)
        for d in os.listdir(EXTERNAL_DRIVE_PATH)
        if os.path.isdir(os.path.join(EXTERNAL_DRIVE_PATH, d))
    ])

    base_port = 8787

    if year_dirs:
        # Load grid information from the first file in the first year directory
        first_ds = nc.Dataset(sorted(glob(os.path.join(year_dirs[0], "*.nc")))[0])
        lon = first_ds.variables['lon'][:] - 360
        lat = first_ds.variables['lat'][:]
        nv = first_ds.variables['nv'][:].T - 1  # Convert from (3, nele) to (nele, 3)
        h = first_ds.variables['h'][:]
        # For sigma layers, extract a representative column (assumes sigma values are the same at all nodes)
        siglay = first_ds.variables['siglay'][:, 0]
        first_ds.close()

        # Compute or load triangle areas using the new projected approach
        if os.path.exists(TRIANGLE_AREAS_PATH):
            triangle_areas = np.load(TRIANGLE_AREAS_PATH)
        else:
            triangle_areas = compute_triangle_areas(lon, lat, nv)

        # Write the grid file (to be used in later spatial analyses)
        write_grid_file(lon, lat, nv, h, triangle_areas)
        logger.info(f"Grid file written to {GRID_FILE}")

        # Process each year's data
        for i, year_dir in enumerate(year_dirs):
            year_label = extract_year(os.path.basename(year_dir))
            client = setup_dask(base_port + i)
            process_fvcom_year(year_dir, year_label, client, lon, lat, nv, h, siglay, triangle_areas)
            client.close()





###### TESTING SINGLE DAY FROM SINGLE FILE
if __name__ == "__main__":
    # Use a single test file from one directory for testing
    #test_file = sorted(glob(os.path.join(EXTERNAL_DRIVE_PATH, "TestYear", "*.nc")))[0]
    # Testfile single path
    test_file = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOM_dataforRnd2/20170530_2016/erie_0008.nc'  # WD path small file 2016"
    logger.info(f"Using test file: {test_file}")

    # Open the test file
    ds = nc.Dataset(test_file, mode="r")

    # Read grid variables from the test file
    lon = ds.variables['lon'][:] - 360
    lat = ds.variables['lat'][:]
    nv = ds.variables['nv'][:].T - 1  # Convert from (3, nele) to (nele, 3)
    h = ds.variables['h'][:]
    siglay = ds.variables['siglay'][:, 0]

    # Compute triangle areas (if needed)
    if os.path.exists(TRIANGLE_AREAS_PATH):
        triangle_areas = np.load(TRIANGLE_AREAS_PATH)
    else:
        triangle_areas = compute_triangle_areas(lon, lat, nv)

    # Optionally write the grid file for testing
    write_grid_file(lon, lat, nv, h, triangle_areas)

    # Process just this file (simulate a single day)
    Times = ds.variables['Times'][:, :]
    time = np.array(["".join(char.decode("utf-8") for char in row).strip() for row in Times])
    time = pd.to_datetime(time, format="%Y-%m-%dT%H:%M:%S.%f", utc=True, errors='coerce')
    mask_valid = ~pd.isnull(time)
    time = time[mask_valid]
    daychar = time.strftime("%Y-%m-%d")
    days = np.unique(daychar)

    # For testing, just process the first day
    test_day = days[0]
    indices = np.where(daychar == test_day)[0]

    temp = ds.variables['temp'][:]  # (time, siglay, node)
    do = ds.variables['Dissolved_oxygen'][:]  # (time, siglay, node)
    zeta = ds.variables['zeta'][:]  # (time, node)

    # Compute daily means for the test day
    temp_day = np.mean(temp[indices, :, :], axis=0).T  # shape (node, siglay)
    do_day = np.mean(do[indices, :, :], axis=0).T  # shape (node, siglay)
    zeta_day = np.mean(zeta[indices, :], axis=0)  # shape (node,)

    # Compute actual depth at each node and layer
    depth = h[:, np.newaxis] * siglay[np.newaxis, :] + zeta_day[:, np.newaxis]
    logger.info(f"Test day depth shape: {depth.shape}")

    ds.close()
# Main execution block
if __name__ == "__main__":
    year_dirs = sorted([
        os.path.join(EXTERNAL_DRIVE_PATH, d)
        for d in os.listdir(EXTERNAL_DRIVE_PATH)
        if os.path.isdir(os.path.join(EXTERNAL_DRIVE_PATH, d))
    ])

    base_port = 8787

    if year_dirs:
        # Load grid information from the first file in the first year directory
        first_ds = nc.Dataset(sorted(glob(os.path.join(year_dirs[0], "*.nc")))[0])
        lon = first_ds.variables['lon'][:] - 360
        lat = first_ds.variables['lat'][:]
        nv = first_ds.variables['nv'][:].T - 1  # Convert from (3, nele) to (nele, 3)
        h = first_ds.variables['h'][:]
        # For sigma layers, extract a representative column (assumes sigma values are the same at all nodes)
        siglay = first_ds.variables['siglay'][:, 0]
        first_ds.close()

        # Compute or load triangle areas using the new projected approach
        if os.path.exists(TRIANGLE_AREAS_PATH):
            triangle_areas = np.load(TRIANGLE_AREAS_PATH)
        else:
            triangle_areas = compute_triangle_areas(lon, lat, nv)

        # Write the grid file (to be used in later spatial analyses)
        write_grid_file(lon, lat, nv, h, triangle_areas)
        logger.info(f"Grid file written to {GRID_FILE}")

        # Process each year's data
        for i, year_dir in enumerate(year_dirs):
            year_label = extract_year(os.path.basename(year_dir))
            client = setup_dask(base_port + i)
            process_fvcom_year(year_dir, year_label, client, lon, lat, nv, h, siglay, triangle_areas)
            client.close()
"""

#%%
# !/usr/bin/env python3
"""
Modified FVCOM processing script.

############# THIS SCRIPT NEEDS SIMPLE MANUAL ADJUSTMENT FOR FILE FILTERING AND USE FROM DIR BETWEEN ####.NC AND ERIE_000#.NC 
        ##### SEE MAIN AND FVCOM PROCESSING +CHECK OTHERS 
        ##### THE ####.NC FILES ARE WEIRD 23 HR, 1 HR FOR SAME DAY AND REQUIRE POST PROCESSING
Key modifications:
  1. Files with basenames starting with 'erie_000' are now filtered out.
  2. The script now loops over all valid files in each year directory and
     aggregates data by unique day (daily averages) before writing output.
  3. The grid file is written conditionally (skipped if it exists unless overwritten).

Note: You may see warnings from the multiprocessing resource_tracker about leaked
semaphore objects. This is a known issue with Python’s multiprocessing and Dask,
and if the output is correct, it can usually be safely ignored.
"""
# LEFH - See the condition for file in file list in process_Fvcom_yeaer and in __main__

import numpy as np
import netCDF4 as nc
import dask.array as da
import os
import re
import logging
import pandas as pd
from dask.distributed import Client, LocalCluster
from glob import glob
import xarray as xr
import pyproj

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Paths and Options
# ---------------------------
EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/Temp_test'
# EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/Temporary_redo'  # Alternative paths commented out below
# EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOM_dataforRnd2'
# EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share'
OUTPUT_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/Temporary_redoconcatdates'
# OUTPUT_PATH  = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2'
TRIANGLE_AREAS_PATH = os.path.join(OUTPUT_PATH, 'triangle_areas.npy')
GRID_FILE = os.path.join(OUTPUT_PATH, 'grid.nc')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Option: Set to True to force overwriting the grid file (e.g., if node attributes change)
OVERWRITE_GRID = False

# ---------------------------
# Dask Cluster Setup Function
# ---------------------------
def setup_dask(port):
    cluster = LocalCluster(
        n_workers=6,
        threads_per_worker=1,
        memory_limit="8GB",
        local_directory="/Volumes/WD Backup/dask_tmp",
        dashboard_address=f":{port}"
    )
    return Client(cluster)

# ---------------------------
# Extract Year from Folder Name
# ---------------------------
def extract_year(folder_name):
    match = re.search(r'_(\d{4})$', folder_name)
    return match.group(1) if match else folder_name

# ---------------------------
# Compute Triangle Areas
# ---------------------------
def compute_triangle_areas(lon, lat, nv):
    logger.info("Computing triangle areas using projected coordinates...")
    # Create a transformer: from EPSG:4326 (lon/lat) to EPSG:32617 (UTM zone 17N, in meters)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    # Transform all node coordinates from degrees to meters
    x, y = transformer.transform(lon, lat)
    # Use the connectivity array (nv) to get the coordinates of each triangle's vertices
    p1 = np.stack((x[nv[:, 0]], y[nv[:, 0]]), axis=1)
    p2 = np.stack((x[nv[:, 1]], y[nv[:, 1]]), axis=1)
    p3 = np.stack((x[nv[:, 2]], y[nv[:, 2]]), axis=1)
    # Compute edge lengths in meters using Euclidean norm
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p3, axis=1)
    c = np.linalg.norm(p3 - p1, axis=1)
    s = (a + b + c) / 2
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    np.save(TRIANGLE_AREAS_PATH, areas)
    return areas

# ---------------------------
# Write Grid File (with option to not overwrite)
# ---------------------------
def write_grid_file(lon, lat, nv, h, triangle_areas):
    logger.info("Writing grid file...")
    ds_grid = nc.Dataset(GRID_FILE, "w", format="NETCDF4")
    ds_grid.createDimension("nele", nv.shape[0])
    ds_grid.createDimension("three", nv.shape[1])
    ds_grid.createDimension("node", lon.shape[0])

    area_var = ds_grid.createVariable("triangle_areas", "f4", ("nele",))
    nv_var = ds_grid.createVariable("nv", "i4", ("nele", "three"))
    lon_var = ds_grid.createVariable("lon", "f4", ("node",))
    lat_var = ds_grid.createVariable("lat", "f4", ("node",))
    h_var = ds_grid.createVariable("h", "f4", ("node",))

    area_var[:] = triangle_areas
    nv_var[:] = nv  # nv should already be transposed and zero-indexed
    lon_var[:] = lon
    lat_var[:] = lat
    h_var[:] = h

    ds_grid.close()

# ---------------------------
# Process a Year's FVCOM NetCDF Files (Aggregating Daily Data from All Valid Files)
# ---------------------------
def process_fvcom_year(year_path, year_label, client, lon, lat, nv, h, siglay, triangle_areas):
    logger.info(f"Processing Year: {year_label} at {year_path}")

    # Get all .nc files in the year directory, then filter out files starting with "erie_000"
    file_list = sorted(glob(os.path.join(year_path, "*.nc")))
    file_list = [f for f in file_list if not os.path.basename(f).startswith("erie_000")]

    if not file_list:
        logger.warning(f"No valid files found in {year_path}. Skipping.")
        return

    # Initialize lists to accumulate results across all files
    all_temp = []  # list for cell temperature for each day (shape: (nele, num_layers))
    all_do = []    # list for cell dissolved oxygen for each day
    all_depth = [] # list for cell depth for each day
    all_dates = [] # list for date strings

    # Loop over each file in the directory
    for f in file_list:
        logger.info(f"Processing file: {f}")
        try:
            ds = nc.Dataset(f, mode="r")
        except Exception as e:
            logger.error(f"Error opening file {f}: {e}", exc_info=True)
            continue

        # Extract time information and convert to datetime
        Times = ds.variables['Times'][:, :]
        time = np.array(["".join(char.decode("utf-8") for char in row).strip() for row in Times])
        time = pd.to_datetime(time, format="%Y-%m-%dT%H:%M:%S.%f", utc=True, errors='coerce')
        mask_valid = ~pd.isnull(time)
        time = time[mask_valid]
        daychar = time.strftime("%Y-%m-%d")
        days = np.unique(daychar)
        logger.info(f"Found {len(days)} unique day(s) in file {os.path.basename(f)}: {days}")

        # Read dynamic variables from the file
        temp = ds.variables['temp'][:]  # shape: (time, siglay, node)
        do = ds.variables['Dissolved_oxygen'][:]  # shape: (time, siglay, node)
        zeta = ds.variables['zeta'][:]  # shape: (time, node)
        h = ds.variables['h'][:]  # shape: (node,)
        siglay = ds.variables['siglay'][:, 0]  # shape: (siglay,)
        ds.close()

        # For each unique day in the file, compute daily means and cell-based averages
        for day in days:
            indices = np.where(daychar == day)[0]
            # Compute daily means; transpose so that node becomes the first dimension
            temp_day = np.mean(temp[indices, :, :], axis=0).T  # becomes (node, siglay)
            do_day = np.mean(do[indices, :, :], axis=0).T  # becomes (node, siglay)
            zeta_day = np.mean(zeta[indices, :], axis=0)  # becomes (node,)

            # Compute actual depth: depth = h * siglay + zeta_day, result shape: (node, siglay)
            depth = h[:, np.newaxis] * siglay[np.newaxis, :] + zeta_day[:, np.newaxis]

            num_layers = siglay.shape[0]
            # Initialize arrays for cell-based averages
            cell_temp_3d = np.zeros((nv.shape[0], num_layers))
            cell_do_3d = np.zeros((nv.shape[0], num_layers))
            cell_depth_3d = np.zeros((nv.shape[0], num_layers))

            for i in range(num_layers):
                layer_depth = depth[:, i]  # (node,)
                # For each cell, average the values from the 3 nodes (using connectivity nv)
                mean_depth = np.mean(layer_depth[nv], axis=1)
                mean_temp = np.mean(temp_day[nv, i], axis=1)
                mean_do = np.mean(do_day[nv, i], axis=1)
                cell_temp_3d[:, i] = mean_temp
                cell_do_3d[:, i] = mean_do
                cell_depth_3d[:, i] = mean_depth

            # Append daily results from this file
            all_temp.append(cell_temp_3d)
            all_do.append(cell_do_3d)
            all_depth.append(cell_depth_3d)
            all_dates.append(day)

    # Stack the lists along a new time axis
    all_temp = np.stack(all_temp, axis=0)  # shape: (total_days, nele, num_layers)
    all_do = np.stack(all_do, axis=0)
    all_depth = np.stack(all_depth, axis=0)

    # Create an xarray Dataset for final output
    ds_out = xr.Dataset(
        {
            "temperature": (("time", "nele", "num_layers"), all_temp),
            "dissolved_oxygen": (("time", "nele", "num_layers"), all_do),
            "depth": (("time", "nele", "num_layers"), all_depth)
        },
        coords={
            "time": all_dates,
            "nele": np.arange(nv.shape[0]),
            "num_layers": np.arange(siglay.shape[0])
        }
    )

    # Write the aggregated dataset as a netCDF file
    output_nc_file = os.path.join(OUTPUT_PATH, f"spatial_summary_{year_label}.nc")
    ds_out.to_netcdf(output_nc_file)
    logger.info(f"Written aggregated netCDF output: {output_nc_file}")

# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == "__main__":
    # Get list of year directories from the external drive
    year_dirs = sorted([
        os.path.join(EXTERNAL_DRIVE_PATH, d)
        for d in os.listdir(EXTERNAL_DRIVE_PATH)
        if os.path.isdir(os.path.join(EXTERNAL_DRIVE_PATH, d))
    ])

    base_port = 8787

    # Use grid information from the first full-domain file in the first year directory that meets the filter criteria
    first_year_dir = year_dirs[0]
    first_file_list = sorted(glob(os.path.join(first_year_dir, "*.nc")))
    first_file_list = [f for f in first_file_list if not os.path.basename(f).startswith("erie_000")]
    first_ds = nc.Dataset(first_file_list[0], mode="r")
    lon = first_ds.variables['lon'][:] - 360
    lat = first_ds.variables['lat'][:]
    nv = first_ds.variables['nv'][:].T - 1  # Ensure nv is (nele, 3)
    h = first_ds.variables['h'][:]
    siglay = first_ds.variables['siglay'][:, 0]  # Use one representative column for sigma layers
    first_ds.close()

    # Compute or load triangle areas using the new projected approach
    if os.path.exists(TRIANGLE_AREAS_PATH):
        triangle_areas = np.load(TRIANGLE_AREAS_PATH)
    else:
        triangle_areas = compute_triangle_areas(lon, lat, nv)

    # Write the grid file only if it doesn't already exist or if overwrite is forced
    if OVERWRITE_GRID or not os.path.exists(GRID_FILE):
        write_grid_file(lon, lat, nv, h, triangle_areas)
        logger.info(f"Grid file written to {GRID_FILE}")
    else:
        logger.info(f"Grid file already exists at {GRID_FILE}. Skipping grid file write.")

    # Process each year's data
    for i, year_dir in enumerate(year_dirs):
        year_label = extract_year(os.path.basename(year_dir))
        client = setup_dask(base_port + i)
        process_fvcom_year(year_dir, year_label, client, lon, lat, nv, h, siglay, triangle_areas)
        client.close()


