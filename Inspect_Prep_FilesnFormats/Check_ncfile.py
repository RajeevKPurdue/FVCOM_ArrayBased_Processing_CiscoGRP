import xarray as xr
import netCDF4 as nc
from netCDF4 import chartostring
import numpy as np
import pandas as pd

#%%
# checking Claude here and chat is below

##### Visual checks and demo plotting code
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# Path to your grid file
GRID_FILE = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/grid.nc'
OUTPUT_FILE = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/spatial_summary_2006.nc'

# Open the grid file
grid_ds = nc.Dataset(GRID_FILE, mode='r')

# Read node coordinates and connectivity
lon_nodes = grid_ds.variables['lon'][:]  # shape: (node,)
lat_nodes = grid_ds.variables['lat'][:]  # shape: (node,)
nv = grid_ds.variables['nv'][:]          # shape: (nele, three)
triangle_areas = grid_ds.variables['triangle_areas'][:]  # shape: (nele,)

# Compute centroids: for each element, average the lon/lat of its 3 vertices
cell_lon = np.mean(lon_nodes[nv], axis=1)  # shape: (nele,)
cell_lat = np.mean(lat_nodes[nv], axis=1)  # shape: (nele,)

grid_ds.close()

print("Cell centroids computed: cell_lon shape =", cell_lon.shape, "cell_lat shape =", cell_lat.shape)

# Open the aggregated netCDF file
ds = xr.open_dataset(OUTPUT_FILE)

# Check the dataset structure
print("Dataset summary:")
print(ds)

# Get the time values
times = ds.time.values
print(f"First 5 timestamps: {times[:5]}")

# Plot the volumetrically-averaged temperature for the first time slice
temp_vol_avg = ds['temperature'][0, :].values  # shape: (nele,)
print("Temperature shape:", temp_vol_avg.shape)

# Plot the depth for the first time slice
depth_vol_avg = ds['depth'][0, :].values  # shape: (nele,)

# Plot dissolved oxygen for the first time slice
do_vol_avg = ds['dissolved_oxygen'][0, :].values  # shape: (nele,)

# Plot the volumetrically-averaged temperature using scatter plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(cell_lon, cell_lat, c=temp_vol_avg, cmap='viridis', s=5)
plt.colorbar(sc, label='Temperature (°C)')
plt.title(f"Volumetrically-Averaged Temperature ({times[0]})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig('temp_vol_avg.png', dpi=300)
plt.show()

# Plot the volumetrically-averaged DO using scatter plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(cell_lon, cell_lat, c=do_vol_avg, cmap='viridis', s=5)
plt.colorbar(sc, label='Dissolved Oxygen (mg/L)')
plt.title(f"Volumetrically-Averaged DO ({times[0]})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig('do_vol_avg.png', dpi=300)
plt.show()

# Create a time series plot for a single element
element_index = 5000  # Choose a specific element (adjust as needed)
time_series_temp = ds['temperature'][:, element_index].values

plt.figure(figsize=(12, 6))
plt.plot(ds.time, time_series_temp)
plt.title(f"Temperature Time Series for Element {element_index}")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.savefig('temp_timeseries.png', dpi=300)
plt.show()

# Create a better visual with polygon geometry
polygons = []
for i in range(min(1000, nv.shape[0])):  # Limit to 1000 polygons for speed
    node_idxs = nv[i, :]
    poly_coords = list(zip(lon_nodes[node_idxs], lat_nodes[node_idxs]))
    polygons.append(Polygon(poly_coords))

# Create subset of data for the limited polygons
temp_subset = temp_vol_avg[:1000]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame({
    'geometry': polygons,
    'temp': temp_subset
}, crs="EPSG:4326")

# Plot with proper coloring
plt.figure(figsize=(12, 10))
ax = gdf.plot(column='temp', cmap='viridis', legend=True,
              legend_kwds={'label': 'Temperature (°C)'})
plt.title(f"Temperature Field with Mesh Elements ({times[0]})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig('temp_polygon.png', dpi=300)
plt.show()


#%%

##### Visual checks and demo plotting code

import xarray as xr
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Path to your grid file
GRID_FILE = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/grid.nc' # '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/grid.nc'

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
ds = xr.open_dataset('/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/spatial_summary_2006.nc') # '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/spatial_summary_2024.nc'

# Check the dataset structure
print(ds)

depth_bottom = ds['depth'][0, :, -1].values  # shape: (nele,)
#depth_bottom = ds['depth'][:, -1]
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
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from tkinter import Tk, filedialog
import os


def select_file(title="Select NetCDF file"):
    """Open a file dialog to select a NetCDF file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


def select_directory(title="Select output directory"):
    """Open a directory dialog to select an output folder"""
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory


def plot_surface_bottom():
    # Ask user to select the NetCDF file
    nc_file = select_file("Select NetCDF file created by the FVCOM script")
    if not nc_file:
        print("No file selected. Exiting.")
        return

    # Ask user to select the grid file (optional)
    grid_file = select_file("Select grid.nc file (Cancel if it's in the same directory as the data file)")
    if not grid_file:
        # Try to find grid file in the same directory
        grid_file = os.path.join(os.path.dirname(nc_file), 'grid.nc')
        print(f"Using grid file: {grid_file}")

    # Ask user to select output directory
    output_dir = select_directory("Select directory to save plots")
    if not output_dir:
        output_dir = os.path.dirname(nc_file)
        print(f"No directory selected. Using {output_dir}")

    # Ask which variable to plot
    variable = input("Which variable to plot? (temperature/dissolved_oxygen) [default: temperature]: ")
    if not variable:
        variable = 'temperature'

    # Open the NetCDF file
    ds = xr.open_dataset(nc_file)

    # Open grid file
    try:
        grid_ds = xr.open_dataset(grid_file)
        element_lon = grid_ds.element_lon.values
        element_lat = grid_ds.element_lat.values
    except:
        print("Could not open grid file. Checking if coordinates are in the data file...")
        if 'element_lon' in ds.variables:
            element_lon = ds.element_lon.values
            element_lat = ds.element_lat.values
        else:
            print("Error: No coordinate information available.")
            return

    # Ask which date to plot
    print("Available dates:")
    dates = pd.to_datetime(ds.time.values)
    for i, date in enumerate(dates):
        print(f"{i}: {date.strftime('%Y-%m-%d')}")

    date_index_str = input(f"Enter date index (0-{len(dates) - 1}) [default: 0]: ")
    if not date_index_str:
        date_index = 0
    else:
        date_index = int(date_index_str)

    date_str = dates[date_index].strftime('%Y-%m-%d')

    # Get variable data
    var_data = ds[variable].values[date_index]
    var_units = ds[variable].attrs.get('units', '')
    var_name = 'Temperature' if variable == 'temperature' else 'Dissolved Oxygen'

    # Get surface and bottom layers
    surface_data = var_data[:, 0]
    bottom_data = var_data[:, -1]

    # Choose colormap
    cmap = cmo.thermal if variable == 'temperature' else cmo.oxy

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    vmin = min(np.nanmin(surface_data), np.nanmin(bottom_data))
    vmax = max(np.nanmax(surface_data), np.nanmax(bottom_data))

    # Plot surface
    sc0 = axes[0].scatter(element_lon, element_lat, c=surface_data,
                          cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.8)
    axes[0].set_title(f'Surface {var_name} - {date_str}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_aspect('equal')

    # Plot bottom
    sc1 = axes[1].scatter(element_lon, element_lat, c=bottom_data,
                          cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.8)
    axes[1].set_title(f'Bottom {var_name} - {date_str}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_aspect('equal')

    # Add colorbar
    cbar = fig.colorbar(sc0, ax=axes, orientation='horizontal', pad=0.05)
    cbar.set_label(f'{var_name} ({var_units})')

    #plt.tight_layout()

    # Save figure
    filename = f"{os.path.basename(nc_file).split('.')[0]}_{date_str}_{variable}_surface_bottom.png"
    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

    plt.show()

    # Close datasets
    ds.close()
    try:
        grid_ds.close()
    except:
        pass


if __name__ == "__main__":
    # Make sure to import pandas here
    import pandas as pd

    plot_surface_bottom()