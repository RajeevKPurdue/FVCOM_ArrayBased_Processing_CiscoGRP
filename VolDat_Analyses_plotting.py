import os
import gc
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# User Parameters
# ---------------------------
# Directory containing processed summary files (one per year)
PROCESSED_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_volumes_Rnd2_AD2000/reiterated_AD_2500'  # CHANGE THIS to your directory
# Output directory for plots (optional)
PLOTS_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_AD2500_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paths to precomputed arrays (triangle areas and management unit mask)
triangle_areas_path = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/triangle_areas.npy'
mu_mask_path = '/Volumes/WD Backup/Rowe_FVCOM_data/mu_mask_line_segments_4MU.npy'

# Resampling scale: use any Pandas offset alias (e.g., "M" for monthly, "3M" for quarterly, etc.)
resample_scale = "M"  # Change as needed

# Threshold on GRP (only cells with average GRP > threshold are counted)
grp_threshold = 0.0

# ---------------------------
# Load Precomputed Arrays
# ---------------------------
if os.path.exists(triangle_areas_path):
    triangle_areas = np.load(triangle_areas_path)
else:
    raise FileNotFoundError(f"Triangle areas file not found at {triangle_areas_path}")

if os.path.exists(mu_mask_path):
    mu_mask = np.load(mu_mask_path)
else:
    raise FileNotFoundError(f"Management unit mask file not found at {mu_mask_path}")

# ---------------------------
# File Filters: List Processed Summary Files
# ---------------------------
# Note: We're filtering for files that start with "grp_" since these are the output files.
processed_files = [
    os.path.join(PROCESSED_DIR, f)
    for f in os.listdir(PROCESSED_DIR)
    if f.startswith("grp_") and f.endswith(".nc")
]


# ---------------------------
# Function Definitions
# ---------------------------
def compute_volume_time_series(ds, threshold=0.0, triangle_areas=None):
    """
    Compute a water column volume time series for cells where the average GRP exceeds the threshold.

    The water column volume for each cell is approximated as:
      cell_volume = triangle_area (per cell) * (sum of vertical layer thicknesses)
    where the vertical layer thicknesses are computed by differencing the "depth" variable along "num_layers".

    Also, the average GRP per cell (over vertical layers) is computed and used as a mask.

    Parameters:
      ds : xarray.Dataset
          Must contain "GRP" and "depth" variables with dimensions (time, nele, num_layers).
      threshold : float
          The threshold value for the average GRP.
      triangle_areas : numpy.ndarray
          1D array of cell areas (length = nele).

    Returns:
      vol_ts : xarray.DataArray
          A time series (indexed by time) of the water column volume (m^3) for cells with GRP > threshold.
      total_vol_ts : xarray.DataArray
          A time series (indexed by time) of the total water column volume (m^3) over all cells.
    """
    # Compute vertical layer thickness: difference in depth along "num_layers"
    # Note: depth values are negative (surface = 0, bottom < 0), so the raw diff is negative.
    dz = ds["depth"].diff(dim="num_layers")
    # For physical thickness, take absolute value
    dz = np.abs(dz)

    # Sum dz over the vertical dimension to get total water column height per cell (dimensions: time, nele)
    dz_sum = dz.sum(dim="num_layers")

    # Create a DataArray for triangle areas (1D, with dimension "nele") and expand to include the "time" dimension.
    if triangle_areas is not None:
        area_da = xr.DataArray(triangle_areas, dims=["nele"])
        area_da = area_da.expand_dims({"time": ds.time})
    else:
        area_da = xr.ones_like(ds["GRP"].isel(time=0).mean(dim="num_layers"))  # Unit area if not provided

    # Compute cell volumes (per time step): (time, nele)
    cell_volumes = area_da * dz_sum

    # Compute the average GRP per cell over vertical layers (dimensions: time, nele)
    grp_cell = ds["GRP"].mean(dim="num_layers")
    # Create a boolean mask where average GRP exceeds threshold
    mask = grp_cell > threshold
    mask_float = mask.astype(float)

    # Compute volume only for cells with positive GRP
    vol_ts = (mask_float * cell_volumes).sum(dim="nele")
    # Also compute total volume over all cells
    total_vol_ts = cell_volumes.sum(dim="nele")
    return vol_ts, total_vol_ts


def resample_time_series(da, scale="M"):
    """
    Resample a DataArray (indexed by time) to the specified scale using the sum.

    Parameters:
      da : xarray.DataArray
          Must have a "time" coordinate.
      scale : str
          A Pandas offset alias (e.g., "M" for monthly, "3M" for quarterly).

    Returns:
      da_resampled : xarray.DataArray
          Resampled time series.
    """
    da["time"] = pd.to_datetime(da["time"].values)
    da_resampled = da.resample(time=scale).sum()
    return da_resampled


def plot_time_series(da, var_type="GRP", title_info="", ylabel="Percent Positive Volume (%)", xlabel="Time",
                     output_path=None):
    """
    Plot a time series DataArray using matplotlib.

    The plot title will include the variable type (e.g. "GRP" or "GRP lethal")
    as well as additional information (e.g., year, iteration, MU) passed via title_info.
    The y-axis shows the percent positive volume over time.

    Parameters:
      da : xarray.DataArray
          The time series to plot.
      var_type : str
          Label to distinguish the variable (e.g., "GRP" or "GRP lethal").
      title_info : str
          Additional information to include in the title.
      ylabel, xlabel : str
          Plot labels.
      output_path : str or None
          If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 6))
    da.plot(marker='o')
    plt.title(f"Percent Positive Volume Time Series ({var_type}) for {title_info}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# ---------------------------
# Main Processing Loop for Volume Time Series
# ---------------------------
# Dictionaries to store overall and MU-specific volume time series
volume_ts_dict = {}  # key: (file_basename, iteration) or file_basename if no iteration
mu_volume_ts_dict = {}  # key: (file_basename, iteration, MU)

for file in processed_files:
    logger.info(f"Processing file: {file}")
    ds = xr.open_dataset(file)
    # Ensure the "time" coordinate is in datetime format
    ds["time"] = pd.to_datetime(ds["time"].values)

    # Check if there is an iteration dimension
    if "iteration" in ds.dims:
        iterations = ds.iteration.values
    else:
        iterations = [None]

    for it in iterations:
        if it is not None:
            ds_iter = ds.sel(iteration=it)
            key_main = (os.path.basename(file), it)
        else:
            ds_iter = ds
            key_main = os.path.basename(file)

        # Convert key_main to a string for the title
        title_info_main = " ".join(map(str, key_main)) if isinstance(key_main, tuple) else key_main

        # Compute overall volume time series (using actual vertical thickness from depth)
        vol_ts, total_vol_ts = compute_volume_time_series(ds_iter, threshold=grp_threshold,
                                                          triangle_areas=triangle_areas)
        # Resample the masked and total volumes separately
        vol_ts_resampled = resample_time_series(vol_ts, scale=resample_scale)
        total_vol_ts_resampled = resample_time_series(total_vol_ts, scale=resample_scale)
        # Compute percent positive volume for each time unit of summary
        vol_percent_resampled = (vol_ts_resampled / total_vol_ts_resampled) * 100
        volume_ts_dict[key_main] = vol_percent_resampled

        # Plot overall time series for GRP
        output_plot = os.path.join(PLOTS_DIR, f"{title_info_main}_percent_volume_GRP.png")
        plot_time_series(vol_percent_resampled, var_type="GRP", title_info=title_info_main, output_path=output_plot)

        # Now, for each management unit (MU), subset along the "nele" dimension using the preloaded mu_mask
        for mu in [1, 2, 3, 4]:
            indices = np.where(mu_mask == mu)[0]
            ds_mu = ds_iter.sel(nele=indices)
            # Subset the triangle areas array accordingly
            triangle_areas_mu = triangle_areas[indices]
            vol_ts_mu, total_vol_ts_mu = compute_volume_time_series(ds_mu, threshold=grp_threshold,
                                                                    triangle_areas=triangle_areas_mu)
            vol_ts_mu_resampled = resample_time_series(vol_ts_mu, scale=resample_scale)
            total_vol_ts_mu_resampled = resample_time_series(total_vol_ts_mu, scale=resample_scale)
            vol_percent_mu_resampled = (vol_ts_mu_resampled / total_vol_ts_mu_resampled) * 100
            if it is not None:
                key_mu = (os.path.basename(file), it, f"MU{mu}")
            else:
                key_mu = (os.path.basename(file), f"MU{mu}")
            mu_volume_ts_dict[key_mu] = vol_percent_mu_resampled

            # Convert key_mu to string for title
            title_info_mu = " ".join(map(str, key_mu)) if isinstance(key_mu, tuple) else key_mu
            #output_plot_mu = os.path.join(PLOTS_DIR, f"{title_info_mu}_percent_volume_GRP.png")
            #plot_time_series(vol_percent_mu_resampled, var_type="GRP", title_info=title_info_mu,
                             #output_path=output_plot_mu)

    ds.close()
    gc.collect()  # Explicitly free memory

logging.info("Volume time series processing and plotting complete.")
# ---------------------------
# BEGIN COMBINED DATAFRAME CREATION SECTION
# ---------------------------
# (Place the new combined dataframe code here)

import re
import seaborn as sns

def extract_year(filename):
    """
    Extract a 4-digit year from a filename.
    """
    m = re.search(r'(\d{4})', filename)
    return m.group(1) if m else "Unknown"

# Create records for full lake data
records_full = []
for key, ts in volume_ts_dict.items():
    if isinstance(key, tuple):
        filename, iteration = key if len(key) >= 2 else (key[0], None)
    else:
        filename, iteration = key, None
    year = extract_year(filename)
    grp_type = "GRP"
    ts_series = ts.to_series()
    for t, val in ts_series.items():
        records_full.append({
            "year": year,
            "iteration": iteration,
            "MU": "Full Lake",
            "GRP_type": grp_type,
            "time": t,
            "percent_positive": val
        })

# Create records for MU-specific data
records_mu = []
for key, ts in mu_volume_ts_dict.items():
    if isinstance(key, tuple):
        if len(key) >= 3:
            filename, iteration, mu_info = key[0], key[1], key[2]
        else:
            filename, mu_info = key[0], key[1]
            iteration = None
    else:
        filename, iteration, mu_info = key, None, "Unknown MU"
    year = extract_year(filename)
    grp_type = "GRP"
    ts_series = ts.to_series()
    for t, val in ts_series.items():
        records_mu.append({
            "year": year,
            "iteration": iteration,
            "MU": mu_info,
            "GRP_type": grp_type,
            "time": t,
            "percent_positive": val
        })

df_full = pd.DataFrame(records_full)
df_mu = pd.DataFrame(records_mu)
df_combined = pd.concat([df_full, df_mu], ignore_index=True)

combined_output_path = os.path.join(PLOTS_DIR, "combined_volume_data.csv")
df_combined.to_csv(combined_output_path, index=False)
logger.info(f"Combined output saved to {combined_output_path}")

# Optionally, create a multiyear box plot using seaborn
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_combined, x="year", y="percent_positive", hue="MU")
plt.title("Multiyear Percent Positive Volume by Management Unit")
plt.xlabel("Year")
plt.ylabel("Percent Positive Volume (%)")
plt.legend(title="MU", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------
# END COMBINED DATAFRAME CREATION SECTION

