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
#EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOM_dataforRnd2'
# EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/Temporary_redo'  # Alternative paths commented out below
EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share'
OUTPUT_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/Temporary_redoconcatdates'
#OUTPUT_PATH  = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025'
TRIANGLE_AREAS_PATH = os.path.join(OUTPUT_PATH, 'triangle_areas.npy')
#GRID_FILE = os.path.join(OUTPUT_PATH, 'grid.nc')
GRID_FILE = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/grid.nc'
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
def write_grid_file(lon, lat, nv, h, siglay, siglev, triangle_areas):
    logger.info("Writing grid file...")
    ds_grid = nc.Dataset(GRID_FILE, "w", format="NETCDF4")
    ds_grid.createDimension("nele", nv.shape[0])
    ds_grid.createDimension("three", nv.shape[1])
    ds_grid.createDimension("node", lon.shape[0])
    ds_grid.createDimension("siglay", siglay.shape[0])
    ds_grid.createDimension("siglev", siglev.shape[0])

    area_var = ds_grid.createVariable("triangle_areas", "f4", ("nele",))
    nv_var = ds_grid.createVariable("nv", "i4", ("nele", "three"))
    lon_var = ds_grid.createVariable("lon", "f4", ("node",))
    lat_var = ds_grid.createVariable("lat", "f4", ("node",))
    h_var = ds_grid.createVariable("h", "f4", ("node",))
    siglay_var = ds_grid.createVariable("siglay", "f4", ("siglay",))
    siglev_var = ds_grid.createVariable("siglev", "f4", ("siglev",))

    # Add element center coordinates (for convenience)
    element_lon = np.mean(lon[nv], axis=1)
    element_lat = np.mean(lat[nv], axis=1)
    element_lon_var = ds_grid.createVariable("element_lon", "f4", ("nele",))
    element_lat_var = ds_grid.createVariable("element_lat", "f4", ("nele",))

    area_var[:] = triangle_areas
    nv_var[:] = nv  # nv should already be transposed and zero-indexed
    lon_var[:] = lon
    lat_var[:] = lat
    h_var[:] = h
    siglay_var[:] = siglay
    siglev_var[:] = siglev
    element_lon_var[:] = element_lon
    element_lat_var[:] = element_lat

    ds_grid.close()


# ---------------------------
# Process a Year's FVCOM NetCDF Files (Aggregating Daily Data from All Valid Files)
# ---------------------------
def process_fvcom_year(year_path, year_label, client, lon, lat, nv, h, siglay, siglev, triangle_areas):
    logger.info(f"Processing Year: {year_label} at {year_path}")

    # Get all .nc files in the year directory, then filter out files starting with "erie_000"
    file_list = sorted(glob(os.path.join(year_path, "*.nc")))
    file_list = [f for f in file_list if not os.path.basename(f).startswith("erie_000")] # lehf - ...'if not'...

    if not file_list:
        logger.warning(f"No valid files found in {year_path}. Skipping.")
        return

    # Number of layers and elements
    num_layers = siglay.shape[0]
    num_levels = siglev.shape[0]
    num_elements = nv.shape[0]

    # We'll accumulate results for each day
    all_dates = []
    all_cell_temps = []
    all_cell_dos = []
    all_cell_depths = []
    all_layer_volumes = []

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
        ds.close()

        # For each unique day in the file, compute daily means and cell-based averages
        for day in days:
            indices = np.where(daychar == day)[0]

            # Compute daily means
            temp_day = np.mean(temp[indices, :, :], axis=0)  # (siglay, node)
            do_day = np.mean(do[indices, :, :], axis=0)  # (siglay, node)
            zeta_day = np.mean(zeta[indices, :], axis=0)  # (node)

            # CORRECTED: Compute actual depths at nodes for each sigma layer and level
            # Shape: (node, siglay)
            node_depths = np.zeros((lon.shape[0], num_layers))
            for i in range(num_layers):
                node_depths[:, i] = (h + zeta_day) * siglay[i]

            # Compute interface depths at nodes using siglev
            # Shape: (node, siglev)
            node_level_depths = np.zeros((lon.shape[0], num_levels))
            for i in range(num_levels):
                node_level_depths[:, i] = (h + zeta_day) * siglev[i]

            # Initialize arrays for cell-based properties
            # Shape: (nele, siglay)
            cell_temp = np.zeros((num_elements, num_layers))
            cell_do = np.zeros((num_elements, num_layers))
            cell_depths = np.zeros((num_elements, num_layers))

            # Shape: (nele, siglev-1) - volumes between sigma levels
            layer_volumes = np.zeros((num_elements, num_levels - 1))

            # For each sigma layer, compute cell-centered values
            for i in range(num_layers):
                # Average node values to get cell-centered values
                # temp_day shape is (siglay, node), so we use temp_day[i, nv]
                # to get values at 3 vertices of each triangle for layer i
                cell_temp[:, i] = np.mean(temp_day[i, nv], axis=1)
                cell_do[:, i] = np.mean(do_day[i, nv], axis=1)
                # node_depths shape is (node, siglay), so we use node_depths[nv, i]
                cell_depths[:, i] = np.mean(node_depths[nv, i], axis=1)

            # Compute volumes of triangular prisms between sigma levels
            for i in range(num_levels - 1):
                # Thickness at each node between adjacent sigma levels
                thickness_upper = node_level_depths[:, i]
                thickness_lower = node_level_depths[:, i + 1]
                thickness_diff = np.abs(thickness_lower - thickness_upper)

                # Average thickness for each element
                element_thickness = np.mean(thickness_diff[nv], axis=1)

                # Volume = area × thickness
                layer_volumes[:, i] = triangle_areas * element_thickness

            # We now have:
            # - cell_temp: temperature at cell center for each sigma layer
            # - cell_do: dissolved oxygen at cell center for each sigma layer
            # - cell_depths: depth at cell center for each sigma layer
            # - layer_volumes: volume of each triangular prism layer

            # Append this day's results
            all_dates.append(day)
            all_cell_temps.append(cell_temp)
            all_cell_dos.append(cell_do)
            all_cell_depths.append(cell_depths)
            all_layer_volumes.append(layer_volumes)

    # Stack the arrays for all days
    if len(all_dates) > 0:
        # Stack along a new time axis
        # Shape: (time, nele, siglay)
        all_cell_temps = np.stack(all_cell_temps, axis=0)
        all_cell_dos = np.stack(all_cell_dos, axis=0)
        all_cell_depths = np.stack(all_cell_depths, axis=0)
        # Shape: (time, nele, siglev-1)
        all_layer_volumes = np.stack(all_layer_volumes, axis=0)

        # Create an xarray Dataset
        ds_out = xr.Dataset(
            {
                "temperature": (("time", "nele", "siglay"), all_cell_temps),
                "dissolved_oxygen": (("time", "nele", "siglay"), all_cell_dos),
                "depth": (("time", "nele", "siglay"), all_cell_depths),
                "layer_volume": (("time", "nele", "siglev_interfaces"), all_layer_volumes)
            },
            coords={
                "time": all_dates,
                "nele": np.arange(num_elements),
                "siglay": siglay,  # Use actual sigma layer values
                "siglev_interfaces": np.arange(num_levels - 1)  # Interface indices
            }
        )

        # Add metadata
        ds_out.temperature.attrs["units"] = "degrees C"
        ds_out.dissolved_oxygen.attrs["units"] = "mg/L"
        ds_out.depth.attrs["units"] = "meters"
        ds_out.layer_volume.attrs["units"] = "cubic meters"

        # Write the aggregated dataset as a netCDF file
        output_nc_file = os.path.join(OUTPUT_PATH, f"spatial_summary_3D_{year_label}.nc")
        ds_out.to_netcdf(output_nc_file)
        logger.info(f"Written 3D aggregated netCDF output: {output_nc_file}")
    else:
        logger.warning(f"No valid data processed for year {year_label}. Skipping output file creation.")


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
    first_file_list = [f for f in first_file_list if not os.path.basename(f).startswith("erie_000")] # lehf - ... if not ...
    first_ds = nc.Dataset(first_file_list[0], mode="r")

    # Extract static grid information
    lon = first_ds.variables['lon'][:] - 360  # Convert from 0-360 to -180-180
    lat = first_ds.variables['lat'][:]
    nv = first_ds.variables['nv'][:].T - 1  # Ensure nv is (nele, 3) and zero-indexed
    h = first_ds.variables['h'][:]
    siglay = first_ds.variables['siglay'][:, 0]  # Use one representative column for sigma layers
    siglev = first_ds.variables['siglev'][:, 0]  # Use one representative column for sigma levels

    first_ds.close()

    # Compute or load triangle areas
    if os.path.exists(TRIANGLE_AREAS_PATH):
        triangle_areas = np.load(TRIANGLE_AREAS_PATH)
    else:
        triangle_areas = compute_triangle_areas(lon, lat, nv)

    # Write the grid file only if it doesn't already exist or if overwrite is forced
    if OVERWRITE_GRID or not os.path.exists(GRID_FILE):
        write_grid_file(lon, lat, nv, h, siglay, siglev, triangle_areas)
        logger.info(f"Grid file written to {GRID_FILE}")
    else:
        logger.info(f"Grid file already exists at {GRID_FILE}. Skipping grid file write.")

    # Process each year's data
    for i, year_dir in enumerate(year_dirs):
        year_label = extract_year(os.path.basename(year_dir))
        client = setup_dask(base_port + i)
        process_fvcom_year(year_dir, year_label, client, lon, lat, nv, h, siglay, siglev, triangle_areas)
        client.close()


#%%
####### VERTICALLY COLLAPSED #########

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
#EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/Temp_test'
# EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/Temporary_redoconcatdates'  # Alternative paths commented out below
EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOM_dataforRnd2'
# EXTERNAL_DRIVE_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share'
OUTPUT_PATH = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025'
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
def write_grid_file(lon, lat, nv, h, siglay, siglev, triangle_areas):
    logger.info("Writing grid file...")
    ds_grid = nc.Dataset(GRID_FILE, "w", format="NETCDF4")
    ds_grid.createDimension("nele", nv.shape[0])
    ds_grid.createDimension("three", nv.shape[1])
    ds_grid.createDimension("node", lon.shape[0])
    ds_grid.createDimension("siglay", siglay.shape[0])
    ds_grid.createDimension("siglev", siglev.shape[0])

    area_var = ds_grid.createVariable("triangle_areas", "f4", ("nele",))
    nv_var = ds_grid.createVariable("nv", "i4", ("nele", "three"))
    lon_var = ds_grid.createVariable("lon", "f4", ("node",))
    lat_var = ds_grid.createVariable("lat", "f4", ("node",))
    h_var = ds_grid.createVariable("h", "f4", ("node",))
    siglay_var = ds_grid.createVariable("siglay", "f4", ("siglay",))
    siglev_var = ds_grid.createVariable("siglev", "f4", ("siglev",))

    area_var[:] = triangle_areas
    nv_var[:] = nv  # nv should already be transposed and zero-indexed
    lon_var[:] = lon
    lat_var[:] = lat
    h_var[:] = h
    siglay_var[:] = siglay
    siglev_var[:] = siglev

    ds_grid.close()


# ---------------------------
# Process a Year's FVCOM NetCDF Files (Aggregating Daily Data from All Valid Files)
# ---------------------------
def process_fvcom_year(year_path, year_label, client, lon, lat, nv, h, siglay, siglev, triangle_areas):
    logger.info(f"Processing Year: {year_label} at {year_path}")

    # Get all .nc files in the year directory, then filter out files starting with "erie_000"
    file_list = sorted(glob(os.path.join(year_path, "*.nc")))
    file_list = [f for f in file_list if os.path.basename(f).startswith("erie_000")] # for lehf dirs - ...if not os.path.basename(f).startswith("erie_000")]

    if not file_list:
        logger.warning(f"No valid files found in {year_path}. Skipping.")
        return

    # Initialize lists to accumulate results across all files
    all_temp = []  # volumetrically-weighted temperature for each day
    all_do = []  # volumetrically-weighted dissolved oxygen for each day
    all_volume = []  # cell volumes for each day
    all_depth = []  # cell depths for each day
    all_dates = []  # list for date strings

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
        ds.close()

        # For each unique day in the file, compute daily means and cell-based averages
        for day in days:
            indices = np.where(daychar == day)[0]
            # Compute daily means
            temp_day = np.mean(temp[indices, :, :], axis=0)  # (siglay, node)
            do_day = np.mean(do[indices, :, :], axis=0)  # (siglay, node)
            zeta_day = np.mean(zeta[indices, :], axis=0)  # (node)

            # Number of sigma layers and elements
            num_layers = siglay.shape[0]
            num_levels = siglev.shape[0]
            num_elements = nv.shape[0]

            # CORRECTED: Compute actual depths at nodes for each sigma layer
            node_depths = np.zeros((lon.shape[0], num_layers))
            for i in range(num_layers):
                # Correct FVCOM sigma coordinate depth calculation
                node_depths[:, i] = (h + zeta_day) * siglay[i]

            # Compute interface depths at nodes using siglev
            node_level_depths = np.zeros((lon.shape[0], num_levels))
            for i in range(num_levels):
                # Same correction for level depths
                node_level_depths[:, i] = (h + zeta_day) * siglev[i]

            # Initialize arrays for cell-based properties
            cell_temp = np.zeros((num_elements, num_layers))
            cell_do = np.zeros((num_elements, num_layers))
            cell_depths = np.zeros((num_elements, num_layers))
            cell_volumes = np.zeros((num_elements, num_layers - 1))  # Prism volumes between layers

            # For each sigma layer, compute cell-centered values
            for i in range(num_layers):
                # Average node values to get cell-centered values
                cell_temp[:, i] = np.mean(temp_day[i, nv], axis=1)
                cell_do[:, i] = np.mean(do_day[i, nv], axis=1)
                cell_depths[:, i] = np.mean(node_depths[nv, i], axis=1)

            # Compute volumes of triangular prisms between sigma layers
            for i in range(num_layers - 1):
                # Calculate thickness between adjacent sigma layers at nodes
                thickness_upper = node_level_depths[:, i]
                thickness_lower = node_level_depths[:, i + 1]
                thickness_diff = np.abs(thickness_lower - thickness_upper)

                # Average the thickness for each element
                element_thickness = np.mean(thickness_diff[nv], axis=1)

                # Volume = area × thickness
                cell_volumes[:, i] = triangle_areas * element_thickness

            # Calculate volumetrically-weighted averages for each cell
            # We'll compute the weighted average across layers for each element
            total_volumes = np.sum(cell_volumes, axis=1)  # Total volume for each element
            weighted_temp = np.zeros(num_elements)
            weighted_do = np.zeros(num_elements)
            weighted_depth = np.zeros(num_elements)

            # Compute volume-weighted averages across sigma layers for each element
            for e in range(num_elements):
                if total_volumes[e] > 0:  # Avoid division by zero
                    # For temperature: average between layers weighted by volume
                    layer_temps = 0.5 * (cell_temp[e, :-1] + cell_temp[e, 1:])  # Average between layers
                    weighted_temp[e] = np.sum(layer_temps * cell_volumes[e]) / total_volumes[e]

                    # For dissolved oxygen: average between layers weighted by volume
                    layer_dos = 0.5 * (cell_do[e, :-1] + cell_do[e, 1:])  # Average between layers
                    weighted_do[e] = np.sum(layer_dos * cell_volumes[e]) / total_volumes[e]

                    # For depth: average between layers weighted by volume
                    layer_depths = 0.5 * (cell_depths[e, :-1] + cell_depths[e, 1:])  # Average between layers
                    weighted_depth[e] = np.sum(layer_depths * cell_volumes[e]) / total_volumes[e]

            # Append daily results
            all_temp.append(weighted_temp)
            all_do.append(weighted_do)
            all_depth.append(weighted_depth)
            all_volume.append(total_volumes)
            all_dates.append(day)

    # Stack the lists along a new time axis
    if all_temp:  # Check that we have at least some valid data
        all_temp = np.stack(all_temp, axis=0)  # shape: (total_days, nele)
        all_do = np.stack(all_do, axis=0)  # shape: (total_days, nele)
        all_depth = np.stack(all_depth, axis=0)  # shape: (total_days, nele)
        all_volume = np.stack(all_volume, axis=0)  # shape: (total_days, nele)

        # Create an xarray Dataset for final output
        ds_out = xr.Dataset(
            {
                "temperature": (("time", "nele"), all_temp),
                "dissolved_oxygen": (("time", "nele"), all_do),
                "depth": (("time", "nele"), all_depth),
                "volume": (("time", "nele"), all_volume)
            },
            coords={
                "time": all_dates,
                "nele": np.arange(nv.shape[0])
            }
        )

        # Write the aggregated dataset as a netCDF file
        output_nc_file = os.path.join(OUTPUT_PATH, f"spatial_summary_{year_label}.nc")
        ds_out.to_netcdf(output_nc_file)
        logger.info(f"Written aggregated netCDF output: {output_nc_file}")
    else:
        logger.warning(f"No valid data processed for year {year_label}. Skipping output file creation.")


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
    first_file_list = [f for f in first_file_list if os.path.basename(f).startswith("erie_000")] # for lefh - ... if not os.path.basename(f).startswith("erie_000")]
    first_ds = nc.Dataset(first_file_list[0], mode="r")

    # Extract static grid information
    lon = first_ds.variables['lon'][:] - 360  # Convert from 0-360 to -180-180
    lat = first_ds.variables['lat'][:]
    nv = first_ds.variables['nv'][:].T - 1  # Ensure nv is (nele, 3) and zero-indexed
    h = first_ds.variables['h'][:]
    siglay = first_ds.variables['siglay'][:, 0]  # Use one representative column for sigma layers

    # Also extract sigma levels (interfaces) - needed for volume calculations
    siglev = first_ds.variables['siglev'][:, 0]  # Use one representative column

    first_ds.close()

    # Compute or load triangle areas
    if os.path.exists(TRIANGLE_AREAS_PATH):
        triangle_areas = np.load(TRIANGLE_AREAS_PATH)
    else:
        triangle_areas = compute_triangle_areas(lon, lat, nv)

    # Write the grid file only if it doesn't already exist or if overwrite is forced
    if OVERWRITE_GRID or not os.path.exists(GRID_FILE):
        write_grid_file(lon, lat, nv, h, siglay, siglev, triangle_areas)
        logger.info(f"Grid file written to {GRID_FILE}")
    else:
        logger.info(f"Grid file already exists at {GRID_FILE}. Skipping grid file write.")

    # Process each year's data
    for i, year_dir in enumerate(year_dirs):
        year_label = extract_year(os.path.basename(year_dir))
        client = setup_dask(base_port + i)
        process_fvcom_year(year_dir, year_label, client, lon, lat, nv, h, siglay, siglev, triangle_areas)
        client.close()