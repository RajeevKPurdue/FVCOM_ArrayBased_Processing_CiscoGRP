"""
This code consists of cells that I have used to check input and output files that have different file structures. It more so shows work and serves as a log.
"""



import xarray as xr
import netCDF4 as nc
from netCDF4 import chartostring
import numpy as np
import pandas as pd


#file_path = "your_file.nc"  # Change to your actual NetCDF file path
larger_testfile_path = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOM_dataforRnd2/20170530_2016/erie_0008.nc' #WD path small file 2016
testfile_path =''
# Open the file - chat wants xr
#ds = xr.open_dataset(file_path)

from netCDF4 import Dataset
# Opening the file that works using dataset and selective reading like a plain ol' .nc file
ds = nc.Dataset(larger_testfile_path)


# Print dataset information
print(ds)

nv_raw = ds.variables['nv'][:]
nv_transposed = nv_raw.T
print("Raw nv shape:", nv_raw.shape)
print("Transposed nv shape:", nv_transposed.shape)

siglay = ds.variables['siglay'][:, 0]
print("siglay shape:", siglay.shape)
print("siglay:", siglay)
#### checking the 'times' variable that is giving me problems
#times_raw = ds.variables['Times'][:]
#string_times = chartostring(times_raw)
#time = np.array([np.datetime64(t.strip()) for t in string_times])
#print(time)

### Testing summarizing to unique days
#days = pd.to_datetime(time).floor('D')
#unique_days = np.unique(days)

### Map day timestamp to respective day
#day_indices = np.searchsorted(unique_days, days)
#print(day_indices)

# Example: compute daily mean for temp
#temp = ds.variables['Temp'][:]

#temp_daily = np.zeros((len(unique_days), temp.shape[1], temp.shape[2]))  # (days, siglay, node)
#for i in range(len(unique_days)):
#    daily_mask = day_indices == i
#    temp_daily[i] = temp[daily_mask].mean(axis=0)

# Print variable names and their dimensions
#for var in ds.variables:
#    print(f"{var}: {ds[var].dims} {ds[var].shape}")
#%%

##### Visual checks and demo plotting code

import xarray as xr
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Path to your grid file
GRID_FILE = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/grid.nc'

# Open the grid file
grid_ds = nc.Dataset(GRID_FILE, mode='r')

# Read node coordinates and connectivity
lon_nodes = grid_ds.variables['lon'][:]  # shape: (node,) e.g., (6106,)
lat_nodes = grid_ds.variables['lat'][:]  # shape: (node,)
nv = grid_ds.variables['nv'][:]            # shape: (nele, three) e.g., (11509, 3)

# Compute centroids: for each element, average the lon/lat of its 3 vertices
cell_lon = np.mean(lon_nodes[nv], axis=1)  # shape: (nele,)
cell_lat = np.mean(lat_nodes[nv], axis=1)  # shape: (nele,)

grid_ds.close()

print("Cell centroids computed: cell_lon shape =", cell_lon.shape, "cell_lat shape =", cell_lat.shape)

# Open the aggregated netCDF file
ds = xr.open_dataset('/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/spatial_summary_2024.nc')

# Check the dataset structure
print(ds)

depth_bottom = ds['depth'][0, :, -1].values  # shape: (nele,)
print("depth var at fixed time, nele:", ds['depth'][0, 0, :].values)
# Extract the temperature for the first time slice and surface layer (layer 0)
temp_surface = ds['temperature'][0, :, 0].values  # shape: (nele,)

print("Temperature surface shape:", temp_surface.shape)

# Plot the surface layer of temperature for the first time slice
# For example, assume "temperature" has dimensions (time, nele, num_layers)
# and you have a mapping of 'nele' (cell index) to spatial coordinates (from your grid file)

# If you have coordinates for each cell (e.g., via your grid file or additional variables),
# you could do a scatter plot. If not, you can simply visualize the raw array as an image.

# Suppose you have x_cell and y_cell arrays for cell centroids
# For example, if you have a grid file with cell centroid coordinates,
# Now that you have the cell centroids (cell_lon and cell_lat) and the corresponding variable values (temp_surface), create a scatter plot.
plt.figure(figsize=(10, 8))
sc = plt.scatter(cell_lon, cell_lat, c=temp_surface, cmap='viridis', s=5)  # s=5 controls marker size
plt.colorbar(sc, label='Temperature')
plt.title("Temperature at Surface Layer (Time Step 0)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

plt.figure(figsize=(10, 8))
sc = plt.scatter(cell_lon, cell_lat, c=depth_bottom, cmap='viridis', s=5)  # s=5 controls marker size
plt.colorbar(sc, label='Temperature')
plt.title("Depth at Bottom Layer (Time Step 0)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
import geopandas as gpd
from shapely.geometry import Polygon

polygons = []
for i in range(nv.shape[0]):
    # Indices of the cell’s 3 nodes
    node_idxs = nv[i, :]
    poly_coords = list(zip(lon_nodes[node_idxs], lat_nodes[node_idxs]))
    polygons.append(Polygon(poly_coords))

gdf = gpd.GeoDataFrame({'geometry': polygons, 'temp': temp_surface}, crs="EPSG:4326")
gdf.plot(column='temp', cmap='viridis', legend=True)
plt.show()

#%%
"CHECKING THE LEFH_SHARE OUTPUT FILES SINCE THEY HAVE DUPLICATE DAYS FROM THE ###..nc files"

import xarray as xr
import pandas as pd
import netCDF4 as nc


# Open a sample output file
ds = xr.open_dataset("/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/spatial_summary_daily_2022.nc")

# Make sure the time coordinate is in datetime format
ds["time"] = pd.to_datetime(ds["time"].values)

# Group by day (using the date part only)
daily_groups = ds.groupby("time.date")

# Print out the count of time steps per day
for day, group in daily_groups:
    count = group["time"].size
    print(f"Outfile -- Date: {day}, Number of time steps: {count}")

"Checking LEFH_SHARE INPUT FILES"

ds_in = (nc.Dataset("/Volumes/WD Backup/Rowe_FVCOM_data/lehf_share/lefh_2017/201715700.nc", mode="r"))

# Convert the Times variable (assumed to be a char array) to a list of strings
times_raw = ds_in.variables["Times"][:]  # e.g., shape (nt, nchar)
times_str = ["".join(t.astype(str)).strip() for t in times_raw]

# Convert to datetime objects
times_infile = pd.to_datetime(times_str, format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce")
print(times_infile[:50])
print("Time differences between consecutive records for single infile:")
print(times_infile.diff().dropna())


#%%

import xarray as xr
import pandas as pd
import numpy as np

# Open the output file and ensure the time coordinate is datetime
ds = xr.open_dataset("/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/spatial_summary_2017.nc")
ds["time"] = pd.to_datetime(ds["time"].values)

# Group by the date part of time
daily_groups = ds.groupby("time.date")

# Loop over each group and check for duplicate (two) time steps
for day, group in daily_groups:
    if group["time"].size == 2:
        print(f"\nChecking duplicate day: {day}")
        for var in ["temperature", "dissolved_oxygen", "depth"]:
            # Extract the two time steps for this variable
            ts0 = group[var].isel(time=0)
            ts1 = group[var].isel(time=1)

            # Check that the shapes (i.e., number of values) match
            if ts0.shape == ts1.shape:
                print(f"  {var}: Shapes match {ts0.shape}")
            else:
                print(f"  {var}: Shape mismatch {ts0.shape} vs {ts1.shape}")

            # Check if all values are equal (using np.allclose to allow for minor rounding differences)
            if np.allclose(ts0.values, ts1.values, equal_nan=True):
                print(f"  {var}: Values are identical.")
            else:
                # Compute difference metrics if needed
                diff = np.abs(ts0.values - ts1.values)
                print(f"  {var}: Values differ! (min diff: {diff.min()}, max diff: {diff.max()})")

#%%
