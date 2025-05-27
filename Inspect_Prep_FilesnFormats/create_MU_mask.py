
"CODE to check shapefiles and create a mask for MUs - only shapefile I found was not compatible with FVCOM grid"

"Checking shapefile"
import geopandas as gpd

# Path to your shapefile – update the path as needed
shp_path = '/Users/rajeevkumar/Documents/Lake_erie_Perch_MU_gisFIles/Lake_Erie_yellow_perch_management_units/Lake_Erie_yellow_perch_management_units.shp'

# Load the shapefile into a GeoDataFrame
shp = gpd.read_file(shp_path)

# Print the GeoDataFrame info to see data types and non-null counts
print("Shapefile Information:")
print(shp.info())

# Print the column names
print("\nColumns in the shapefile:")
print(shp.columns)

# Print the first 5 rows of the GeoDataFrame to inspect sample values
print("\nFirst 5 rows of the shapefile:")
print(shp.head())

# If you expect a specific column (for example, 'MU'), check its unique values:
if 'MU' in shp.columns:
    print("\nUnique values in the 'MU' column:")
    print(shp['MU'].unique())
else:
    print("\nThe 'MU' column was not found. Available columns:")
    print(shp.columns)
#%%

"""
Script: Create a 4-MU Mask using a Line-Segment (Side-of-Line) Approach

Four management units are defined by three boundary lines:
    - MU1: Grid cell centroid is left of the first line ("line_MU1e_MU2w").
    - MU2: Centroid is right of the first line and left of the second line ("line_MU2e_MU3w").
    - MU3: Centroid is right of the second line and left of the third line ("line_MU3w_MU4e").
    - MU4: Centroid is right of the third line.

The script reads the grid file, computes element centroids, assigns an MU label based on the side-of-line tests,
saves the MU mask as a NumPy file, and plots the result for verification.
"""

import os
import numpy as np
import netCDF4 as nc
import logging
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Optionally import geopandas if needed later
try:
    import geopandas as gpd
except ImportError:
    gpd = None

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import os
import numpy as np
import netCDF4 as nc
import logging
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###############################################################################
# 1) Helper functions
###############################################################################

def dms_to_decimal(degrees, minutes, seconds, direction):
    """
    Convert degrees, minutes, seconds to decimal degrees.
    If direction is 'S' or 'W', the result is negative.
    """
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if direction.upper() in ['S', 'W']:
        decimal *= -1
    return decimal

def find_nearest_node(lon_nodes, lat_nodes, target_lon, target_lat):
    """
    Given arrays of node coordinates (lon_nodes, lat_nodes) and a target point
    (target_lon, target_lat), return:
       idx_min: index of the nearest node
       lon_min, lat_min: the coordinates of that nearest node
       dist_min: approximate distance in degrees between that node and the target
    """
    dist_sq = (lon_nodes - target_lon)**2 + (lat_nodes - target_lat)**2
    idx_min = np.argmin(dist_sq)
    dist_min = np.sqrt(dist_sq[idx_min])
    return idx_min, lon_nodes[idx_min], lat_nodes[idx_min], dist_min

def side_of_line(pt, line_start, line_end):
    """
    Determine on which side of the line (from line_start to line_end) the point pt falls.
    Returns a positive value if pt is to the left, negative if to the right, zero if on the line.
    """
    x, y = pt
    x1, y1 = line_start
    x2, y2 = line_end
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


# ---------------------------
# Define Boundary Lines for the 4 MUs
# ---------------------------
# These lines are defined using two endpoints (in decimal degrees).
# The endpoints are converted from DMS.

# First boundary line: Separates MU1 from MU2.
line_MU1e_MU2w = (
    (dms_to_decimal(82, 32, 35, 'W'), dms_to_decimal(41, 23, 46, 'N')),  # Eastern boundary point (US side)
    (dms_to_decimal(82, 26, 40, 'W'), dms_to_decimal(42, 4, 59, 'N'))  # Eastern boundary point (Canadian side)
)
print(line_MU1e_MU2w)
# Second boundary line: Separates MU2 from MU3.
line_MU2e_MU3w = (
    (dms_to_decimal(81, 22, 40, 'W'), dms_to_decimal(41, 42, 40, 'N')),  # New east US corner for MU2
    (dms_to_decimal(81, 41, 50, 'W'), dms_to_decimal(42, 26, 5, 'N'))  # New east Canadian corner for MU2
)
print(line_MU2e_MU3w)
# Third boundary line: Separates MU3 from MU4.
line_MU3w_MU4e = (
    (dms_to_decimal(80, 9, 22, 'W'), dms_to_decimal(42, 6, 43, 'N')),  # Example point along MU3 boundary
    (dms_to_decimal(80, 25, 55, 'W'), dms_to_decimal(42, 34, 30, 'N'))  # Example point along MU4 boundary
)
print(line_MU3w_MU4e)



# ---------------------------
# Load Grid Information and Compute Element Centroids
# ---------------------------
# Path to your FVCOM grid file (grid.nc)
GRID_FILE = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/grid.nc'
logger.info(f"Reading grid file: {GRID_FILE}")
grid_ds = nc.Dataset(GRID_FILE, mode="r")
lon_nodes = grid_ds.variables['lon'][:]  # shape: (n_node,)
lat_nodes = grid_ds.variables['lat'][:]  # shape: (n_node,)
# Get connectivity array (nv); assumed to be 1-indexed. Convert to 0-indexed.
nv = grid_ds.variables['nv'][:]  # shape: (nele, 3)
grid_ds.close()
nv = nv - 1  # convert to 0-indexed if necessary

# Compute element centroids by averaging the coordinates of the three nodes.
centroid_lon = np.mean(lon_nodes[nv], axis=1)
centroid_lat = np.mean(lat_nodes[nv], axis=1)

n_elements = len(centroid_lon)
logger.info(f"Number of elements: {n_elements}")

###############################################################################
# 4) Snap each line endpoint to the nearest node
###############################################################################

def snap_line_endpoints(lon_nodes, lat_nodes, line_coords):
    """
    line_coords is a tuple of two (lon, lat) endpoints.
    Returns a new line_coords where each endpoint is replaced by the nearest node in the grid.
    """
    (lon1, lat1), (lon2, lat2) = line_coords
    idx1, snapped_lon1, snapped_lat1, dist1 = find_nearest_node(lon_nodes, lat_nodes, lon1, lat1)
    idx2, snapped_lon2, snapped_lat2, dist2 = find_nearest_node(lon_nodes, lat_nodes, lon2, lat2)

    logger.info(f"Snapped endpoint 1 from ({lon1:.4f}, {lat1:.4f}) "
                f"to node index={idx1} at ({snapped_lon1:.4f}, {snapped_lat1:.4f}), dist={dist1:.6f} deg")
    logger.info(f"Snapped endpoint 2 from ({lon2:.4f}, {lat2:.4f}) "
                f"to node index={idx2} at ({snapped_lon2:.4f}, {snapped_lat2:.4f}), dist={dist2:.6f} deg")

    return (snapped_lon1, snapped_lat1), (snapped_lon2, snapped_lat2)

# Snap each boundary line
line_MU1e_MU2w_snapped = snap_line_endpoints(lon_nodes, lat_nodes, line_MU1e_MU2w)
line_MU2e_MU3w_snapped = snap_line_endpoints(lon_nodes, lat_nodes, line_MU2e_MU3w)
line_MU3w_MU4e_snapped = snap_line_endpoints(lon_nodes, lat_nodes, line_MU3w_MU4e)



# ---------------------------
# Create the MU Mask: Assign Each Grid Cell an MU Label Using the Line-Segment Approach
# ---------------------------


mu_mask = np.zeros(n_elements, dtype=int)  # Initialize mask; 0 means not assigned

for i in range(n_elements):
    pt = (centroid_lon[i], centroid_lat[i])
    s1 = side_of_line(pt, *line_MU1e_MU2w_snapped) # *line_MU1e_MU2w
    s2 = side_of_line(pt, *line_MU2e_MU3w_snapped) # *line_MU2e_MU3w)
    s3 = side_of_line(pt, *line_MU3w_MU4e_snapped) # *line_MU3w_MU4e)

    # MU1:  "left" of line1 => s1 < 0
    if s1 < 0:
        mu_mask[i] = 1

    # MU2:  "right" of line1 => s1 >= 0, but "left" of line2 => s2 < 0
    elif s1 >= 0 and s2 < 0:
        mu_mask[i] = 2

    # MU3:  "right" of line2 => s2 >= 0, but "left" of line3 => s3 < 0
    elif s2 >= 0 and s3 < 0:
        mu_mask[i] = 3

    # MU4:  "right" of line3 => s3 >= 0
    else:
        mu_mask[i] = 4


# ---------------------------
# Save the MU Mask as a NumPy File
# ---------------------------
mu_mask_file = 'mu_mask_line_segments_4MU.npy'
np.save(mu_mask_file, mu_mask)
logger.info(f"Saved MU mask to {mu_mask_file}")

# ---------------------------
# (Optional) Plot the MU Mask for Verification
# ---------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(centroid_lon, centroid_lat, c=mu_mask, cmap='jet', s=5, marker='o')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid Element Centroids by Management Unit (Line-Segment Approach)')
plt.colorbar(sc, label='MU Label')

# Plot each boundary line for visual reference.
for line, label in zip([line_MU1e_MU2w, line_MU2e_MU3w, line_MU3w_MU4e],
                       ['Line MU1e_MU2w', 'Line MU2e_MU3w', 'Line MU3w_MU4e']):
    x_vals = [line[0][0], line[1][0]]
    y_vals = [line[0][1], line[1][1]]
    plt.plot(x_vals, y_vals, color='black', linewidth=2, label=label)

plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# 6) Plot snapped points for verification
###############################################################################

plt.figure(figsize=(10, 7))
sc = plt.scatter(centroid_lon, centroid_lat, c=mu_mask, cmap='tab10', s=5, marker='o')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('FVCOM Grid Element Centroids by Management Unit (Line Segment + Snapping)')
plt.colorbar(sc, label='MU Label')

# Plot snapped lines in black
for (lon1, lat1), (lon2, lat2) in [line_MU1e_MU2w_snapped, line_MU2e_MU3w_snapped, line_MU3w_MU4e_snapped]:
    plt.plot([lon1, lon2], [lat1, lat2], color='black', linewidth=2)

# Also show each snapped endpoint with a distinct marker
all_snapped_endpoints = [
    line_MU1e_MU2w_snapped[0], line_MU1e_MU2w_snapped[1],
    line_MU2e_MU3w_snapped[0], line_MU2e_MU3w_snapped[1],
    line_MU3w_MU4e_snapped[0], line_MU3w_MU4e_snapped[1],
]
x_end = [pt[0] for pt in all_snapped_endpoints]
y_end = [pt[1] for pt in all_snapped_endpoints]
plt.scatter(x_end, y_end, c='red', marker='x', s=100, label='Snapped Endpoints')

plt.legend()
plt.tight_layout()
plt.show()

logger.info("Script complete.")
