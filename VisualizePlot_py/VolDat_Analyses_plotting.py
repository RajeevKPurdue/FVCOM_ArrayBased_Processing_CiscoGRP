# Most recent spatially explicit volume computation
import os
import gc
import xarray as xr
import numpy as np
import pandas as pd
import re
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
PROCESSED_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_volumes_Claude_AD2512'
# Output directory for plots and tables
PLOTS_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_plots_042025'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paths to precomputed arrays (triangle areas and management unit mask)
triangle_areas_path = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/triangle_areas.npy'
mu_mask_path = '/Volumes/WD Backup/Rowe_FVCOM_data/mu_mask_line_segments_4MU.npy'

# Resampling scale: use any Pandas offset alias (e.g., "M" for monthly, "3M" for quarterly, etc.)
resample_scale = "ME"  # End of month

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
processed_files = [
    os.path.join(PROCESSED_DIR, f)
    for f in os.listdir(PROCESSED_DIR)
    if f.startswith("grp_") and f.endswith(".nc")
]


def compute_volume_time_series(ds, var_name, threshold=0.0, triangle_areas=None):
    """
    Compute water column volume where each layer's GRP exceeds the threshold,
    preserving the vertical dimension in the calculation.

    Parameters:
      ds : xarray.Dataset
          Must contain var_name, "depth", and optionally "layer_volume"
      var_name : str
          Name of the variable to use for thresholding (e.g., "GRP" or "GRP_lethal")
      threshold : float
          The threshold value for GRP.
      triangle_areas : numpy.ndarray
          1D array of cell areas (length = nele).

    Returns:
      vol_ts : xarray.DataArray
          A time series of the water column volume (m^3) for cells and layers with var_name > threshold.
      total_vol_ts : xarray.DataArray
          A time series of the total water column volume (m^3) over all cells.
    """
    if var_name not in ds:
        raise ValueError(f"Variable {var_name} not found in dataset")

    # Get the dimensions from the variable
    var_dims = ds[var_name].dims

    # Identify dimension names
    time_dim = [dim for dim in var_dims if 'time' in dim.lower()][0]
    nele_dim = [dim for dim in var_dims if 'nele' in dim.lower()][0]
    layer_dim = [dim for dim in var_dims if dim not in [time_dim, nele_dim]][0]

    logger.info(f"Dimensions identified: time={time_dim}, nele={nele_dim}, layer={layer_dim}")

    # Calculate layer volumes
    if "layer_volume" in ds:
        logger.info("Using layer_volume from dataset")
        layer_volumes = ds["layer_volume"]
    else:
        logger.info("Computing layer volumes from depth")
        # Get depth values
        if "depth" not in ds:
            raise ValueError("Either layer_volume or depth must be in the dataset")

        # Compute layer thickness from depth differences
        # Assuming depth is negative from surface (0) to bottom
        depth = ds["depth"]

        # We need to calculate thicknesses between layers
        # This requires computing the differences between adjacent depth values
        # If depth is arranged from surface to bottom, diff will be negative,
        # so we take the absolute value
        dz = abs(depth.diff(dim=layer_dim))

        # We may need to pad with a zero layer at the top or bottom depending on
        # how the depths are ordered in the dataset
        if depth[{layer_dim: 0}].mean() > depth[{layer_dim: 1}].mean():
            # Depths decrease with index (surface to bottom)
            padding = xr.zeros_like(dz.isel({layer_dim: 0})).expand_dims(layer_dim)
            dz = xr.concat([padding, dz], dim=layer_dim)
        else:
            # Depths increase with index (bottom to surface)
            padding = xr.zeros_like(dz.isel({layer_dim: -1})).expand_dims(layer_dim)
            dz = xr.concat([dz, padding], dim=layer_dim)

        # Create a DataArray for triangle areas
        if triangle_areas is not None:
            area_da = xr.DataArray(triangle_areas, dims=[nele_dim])
            # Expand to match dimensions of dz
            for dim in dz.dims:
                if dim != nele_dim and dim not in area_da.dims:
                    area_da = area_da.expand_dims({dim: dz[dim]})
        else:
            # Use unit area if triangle_areas not provided
            area_da = xr.ones_like(dz.isel({time_dim: 0}))

        # Compute layer volumes as area × thickness
        layer_volumes = area_da * dz

    # Total volume (sum over all layers)
    total_vol_ts = layer_volumes.sum(dim=[nele_dim, layer_dim])

    # Create mask where GRP > threshold (preserving all dimensions)
    mask = ds[var_name] > threshold
    mask_float = mask.astype(float)

    # Compute volume only for cells with GRP > threshold (for each layer)
    positive_volumes = mask_float * layer_volumes

    # Sum over all elements and layers to get positive volume time series
    vol_ts = positive_volumes.sum(dim=[nele_dim, layer_dim])

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
    # Ensure time is datetime
    time_dim = [dim for dim in da.dims if 'time' in dim.lower()][0]
    da[time_dim] = pd.to_datetime(da[time_dim].values)

    # Resample using the identified time dimension
    da_resampled = da.resample({time_dim: scale}).sum()
    return da_resampled


def plot_percent_volume_boxplot(df, output_path=None):
    """
    Create boxplots of percent positive volume by month, separated by GRP type.

    Parameters:
        df : pandas.DataFrame
            DataFrame with columns: month, GRP_type, percent_positive
        output_path : str
            Path to save the figure
    """
    plt.figure(figsize=(14, 8))

    # Create boxplot using seaborn
    ax = sns.boxplot(x='month', y='percent_positive', hue='GRP_type', data=df)

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title('Percent Volume with Positive GRP Values by Month')
    plt.xlabel('Month')
    plt.ylabel('Percent Positive Volume (%)')
    plt.legend(title='GRP Type')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.close()


def plot_combined_barplot(df, group_by='year', hue='MU', output_path=None):
    """
    Create a bar plot of percent positive volume.

    Parameters:
        df : pandas.DataFrame
            DataFrame with the columns specified in group_by and hue
        group_by : str
            Column to group by on the x-axis
        hue : str
            Column to use for the hue separation
        output_path : str
            Path to save the figure
    """
    plt.figure(figsize=(14, 8))

    # Calculate mean values for each group
    summary_df = df.groupby([group_by, hue])['percent_positive'].mean().reset_index()

    # Create bar plot
    ax = sns.barplot(x=group_by, y='percent_positive', hue=hue, data=summary_df)

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title(f'Average Percent Volume with Positive GRP Values by {group_by}')
    plt.xlabel(group_by.capitalize())
    plt.ylabel('Percent Positive Volume (%)')
    plt.legend(title=hue)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.close()


def extract_parameters_from_filename(filename):
    """
    Extract mass and P coefficients from filename like "grp_m200_p0.35_*.nc"

    Returns:
        tuple: (mass, p_value)
    """
    mass_match = re.search(r'm(\d+)', filename)
    p_match = re.search(r'p([\d\.]+)', filename)

    mass = mass_match.group(1) if mass_match else "unknown"
    p_value = p_match.group(1) if p_match else "unknown"

    return mass, p_value


def extract_year(filename):
    """
    Extract a 4-digit year from a filename.
    """
    m = re.search(r'(\d{4})', filename)
    return m.group(1) if m else "Unknown"


# ---------------------------
# Main Processing Loop for Volume Time Series
# ---------------------------
# Dictionaries to store results for different GRP types, MUs, and parameter sets
records = []

for file in processed_files:
    filename = os.path.basename(file)
    logger.info(f"Processing file: {filename}")

    # Extract parameters from filename
    mass, p_value = extract_parameters_from_filename(filename)
    year = extract_year(filename)

    # Create a parameter set identifier
    param_set = f"m{mass}_p{p_value}"

    try:
        ds = xr.open_dataset(file)

        # Get time dimension name
        time_dim = [dim for dim in ds.dims if 'time' in dim.lower()][0]

        # Ensure the time coordinate is in datetime format
        ds[time_dim] = pd.to_datetime(ds[time_dim].values)

        # Process both GRP and GRP_lethal if available
        for var_name in ["GRP", "GRP_lethal"]:
            if var_name not in ds:
                logger.warning(f"{var_name} not found in {filename}, skipping")
                continue

            logger.info(f"Processing {var_name} for {param_set} in {year}")

            # First process the full lake
            vol_ts, total_vol_ts = compute_volume_time_series(
                ds, var_name=var_name, threshold=grp_threshold, triangle_areas=triangle_areas
            )

            # Resample both time series
            vol_ts_resampled = resample_time_series(vol_ts, scale=resample_scale)
            total_vol_ts_resampled = resample_time_series(total_vol_ts, scale=resample_scale)

            # Calculate percent positive volume
            vol_percent_resampled = (vol_ts_resampled / total_vol_ts_resampled) * 100

            # Convert to pandas Series for easier handling
            vol_percent_series = vol_percent_resampled.to_series()

            # Add to records for the full lake
            for date, value in vol_percent_series.items():
                month = date.strftime('%b')  # Abbreviated month name
                month_num = date.month
                records.append({
                    'year': year,
                    'month': month,
                    'month_num': month_num,
                    'date': date,
                    'MU': 'Full Lake',
                    'GRP_type': var_name,
                    'param_set': param_set,
                    'mass': mass,
                    'p_value': p_value,
                    'percent_positive': value
                })

            # Now process each management unit
            for mu in [1, 2, 3, 4]:
                mu_name = f"MU{mu}"
                logger.info(f"Processing {var_name} for {param_set} in {year} for {mu_name}")

                # Get indices for this MU
                indices = np.where(mu_mask == mu)[0]

                # Check if indices exist in the dataset's nele dimension
                nele_dim = [dim for dim in ds.dims if 'nele' in dim.lower()][0]
                if max(indices) >= ds.dims[nele_dim]:
                    logger.warning(f"MU{mu} indices exceed nele dimension in {filename}, skipping")
                    continue

                # Subset the dataset and triangle areas for this MU
                ds_mu = ds.isel({nele_dim: indices})
                triangle_areas_mu = triangle_areas[indices]

                # Compute volume time series for this MU
                vol_ts_mu, total_vol_ts_mu = compute_volume_time_series(
                    ds_mu, var_name=var_name, threshold=grp_threshold, triangle_areas=triangle_areas_mu
                )

                # Resample
                vol_ts_mu_resampled = resample_time_series(vol_ts_mu, scale=resample_scale)
                total_vol_ts_mu_resampled = resample_time_series(total_vol_ts_mu, scale=resample_scale)

                # Calculate percent positive volume
                vol_percent_mu_resampled = (vol_ts_mu_resampled / total_vol_ts_mu_resampled) * 100

                # Convert to pandas Series
                vol_percent_mu_series = vol_percent_mu_resampled.to_series()

                # Add to records for this MU
                for date, value in vol_percent_mu_series.items():
                    month = date.strftime('%b')  # Abbreviated month name
                    month_num = date.month
                    records.append({
                        'year': year,
                        'month': month,
                        'month_num': month_num,
                        'date': date,
                        'MU': mu_name,
                        'GRP_type': var_name,
                        'param_set': param_set,
                        'mass': mass,
                        'p_value': p_value,
                        'percent_positive': value
                    })

        ds.close()

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        continue

    gc.collect()  # Free memory

# ---------------------------
# Create DataFrame and Generate Plots
# ---------------------------
df = pd.DataFrame(records)

# Save the complete table
output_table_path = os.path.join(PLOTS_DIR, "complete_GRP_volume_data.csv")
df.to_csv(output_table_path, index=False)
logger.info(f"Complete data table saved to {output_table_path}")

# Generate boxplots for each parameter set and MU
for param_set in df['param_set'].unique():
    # Filter for this parameter set
    df_param = df[df['param_set'] == param_set]

    # 1. Full Lake boxplot by month
    df_full = df_param[df_param['MU'] == 'Full Lake']
    plot_percent_volume_boxplot(
        df_full,
        output_path=os.path.join(PLOTS_DIR, f"{param_set}_full_lake_monthly_boxplot.png")
    )

    # 2. Each MU boxplot by month
    for mu in df_param['MU'].unique():
        if mu == 'Full Lake':
            continue

        df_mu = df_param[df_param['MU'] == mu]
        plot_percent_volume_boxplot(
            df_mu,
            output_path=os.path.join(PLOTS_DIR, f"{param_set}_{mu}_monthly_boxplot.png")
        )

    # 3. Comparison across MUs (bar plot)
    plot_combined_barplot(
        df_param,
        group_by='month',
        hue='MU',
        output_path=os.path.join(PLOTS_DIR, f"{param_set}_MU_comparison_barplot.png")
    )

    # 4. Comparison of GRP vs GRP_lethal (bar plot)
    plot_combined_barplot(
        df_param[df_param['MU'] == 'Full Lake'],
        group_by='month',
        hue='GRP_type',
        output_path=os.path.join(PLOTS_DIR, f"{param_set}_GRP_types_comparison_barplot.png")
    )

# Generate multiyear comparison for each MU
for mu in df['MU'].unique():
    df_mu_all = df[df['MU'] == mu]

    # Bar plot of years by GRP type
    plot_combined_barplot(
        df_mu_all,
        group_by='year',
        hue='GRP_type',
        output_path=os.path.join(PLOTS_DIR, f"{mu}_yearly_GRP_comparison_barplot.png")
    )

    # Bar plot of parameter sets
    plot_combined_barplot(
        df_mu_all,
        group_by='param_set',
        hue='GRP_type',
        output_path=os.path.join(PLOTS_DIR, f"{mu}_parameter_comparison_barplot.png")
    )

logger.info("Processing complete. All plots and tables have been saved.")



#######################
#######################
#########################

#######################
#######################
#########################

#%%
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
PROCESSED_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_volumes_Rnd2_AD2000/reiterated_AD_2512'  # CHANGE THIS to your directory
# Output directory for plots (optional)
PLOTS_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/voldattryagain202504' #'/Volumes/WD Backup/Rowe_FVCOM_data/GRP_AD2500_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paths to precomputed arrays (triangle areas and management unit mask)
triangle_areas_path = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2/triangle_areas.npy'
mu_mask_path = '/Volumes/WD Backup/Rowe_FVCOM_data/mu_mask_line_segments_4MU.npy'

# Resampling scale: use any Pandas offset alias (e.g., "M" for monthly, "3M" for quarterly, etc.)
resample_scale = "ME"  # Change as needed

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
        filename = key[0]
        iteration = str(key[1]) if len(key) >= 2 else None  # Convert iteration to string
    else:
        filename = key
        iteration = None
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
            filename, iteration, mu_info = key[0], str(key[1]), key[2]  # Convert iteration to string
        elif len(key) == 2:
            # Check if the second element is a MU info or an iteration
            if isinstance(key[1], str) and key[1].startswith("MU"):
                filename, mu_info = key[0], key[1]
                iteration = None
            else:
                filename, iteration = key[0], str(key[1])
                mu_info = "Unknown MU"
        else:
            filename = key[0]
            iteration = None
            mu_info = "Unknown MU"
    else:
        filename = key
        iteration = None
        mu_info = "Unknown MU"
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

#%%

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

#%%



# Add required imports if not already present
import os
import gc
import numpy as np
import xarray as xr
import pandas as pd
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------
# User Parameters
# ---------------------------
# Directory containing processed summary files (one per year)
PROCESSED_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_volumes_Claude_AD2512'
# Output directory for plots and tables
PLOTS_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_plots_042025/reiterated_empty'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paths to precomputed arrays (triangle areas and management unit mask)
triangle_areas_path = '/Volumes/WD Backup/Rowe_FVCOM_data/FVCOMvol_Claude_042025/triangle_areas.npy'
mu_mask_path = '/Volumes/WD Backup/Rowe_FVCOM_data/mu_mask_line_segments_4MU.npy'

# Resampling scale: use any Pandas offset alias (e.g., "M" for monthly, "3M" for quarterly, etc.)
resample_scale = "ME"  # End of month

# Threshold on GRP (only cells with GRP > threshold are counted)
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
processed_files = [
    os.path.join(PROCESSED_DIR, f)
    for f in os.listdir(PROCESSED_DIR)
    if f.startswith("grp_") and f.endswith(".nc")
]


# ---------------------------
# Function Definitions
# ---------------------------
def compute_volume_time_series(ds, var_name, threshold=0.0):
    """
    Compute water column volume where each layer's GRP exceeds the threshold,
    accounting for the structure of FVCOM grid data.

    Parameters:
      ds : xarray.Dataset
          Dataset containing the variables
      var_name : str
          Name of the variable to use for thresholding (e.g., "GRP" or "GRP_lethal")
      threshold : float
          The threshold value for GRP.

    Returns:
      vol_ts : xarray.DataArray
          A time series of the water column volume (m^3) where var_name > threshold.
      total_vol_ts : xarray.DataArray
          A time series of the total water column volume (m^3).
    """
    if var_name not in ds:
        raise ValueError(f"Variable {var_name} not found in dataset")

    # Get dimension names
    time_dim = "time"
    nele_dim = "nele"

    # GRP is defined on siglay layers
    siglay_dim = "siglay"

    # layer_volume is defined between siglev interfaces
    vol_dim = "siglev_interfaces"

    logger.info(f"Processing {var_name} with dimensions: time={time_dim}, nele={nele_dim}, layer={siglay_dim}")

    # Verify layer_volume is in the dataset
    if "layer_volume" not in ds:
        raise ValueError("layer_volume not found in dataset")

    logger.info(f"Layer volume dimension: {vol_dim}")

    # Get values as numpy arrays for efficiency
    grp_values = ds[var_name].values  # Shape: (time, nele, siglay)
    volumes = ds["layer_volume"].values  # Shape: (time, nele, siglev_interfaces)

    # Get array shapes
    n_times = grp_values.shape[0]
    n_elements = grp_values.shape[1]
    n_layers = grp_values.shape[2]  # Number of siglay layers
    n_interfaces = volumes.shape[2]  # Number of layer interfaces

    logger.info(f"Array shapes: GRP {grp_values.shape}, volumes {volumes.shape}")

    # Create arrays to store results
    positive_volumes = np.zeros(n_times)
    total_volumes = np.zeros(n_times)

    # Process each time step
    for t in range(n_times):
        # Calculate total volume for this time step
        total_volumes[t] = np.sum(volumes[t, :, :])

        # Calculate positive volume
        # Each siglay layer corresponds to an interface in layer_volume
        # IMPORTANT: The layer_volume at index i corresponds to the volume between
        # siglev interfaces i and i+1, which is centered at siglay i

        # For each element in the lake
        positive_vol_time_t = 0
        for n in range(n_elements):
            # For each sigma layer
            for l in range(n_layers):
                # Check if GRP > threshold for this layer
                if l < n_interfaces and grp_values[t, n, l] > threshold:
                    # Count this layer's volume
                    positive_vol_time_t += volumes[t, n, l]

        positive_volumes[t] = positive_vol_time_t

    # Convert to xarray DataArrays
    vol_ts = xr.DataArray(
        positive_volumes,
        dims=[time_dim],
        coords={time_dim: ds[time_dim]}
    )

    total_vol_ts = xr.DataArray(
        total_volumes,
        dims=[time_dim],
        coords={time_dim: ds[time_dim]}
    )

    return vol_ts, total_vol_ts


def resample_time_series(da, scale="M"):
    """
    Resample a DataArray (indexed by time) to the specified scale using the sum.
    """
    # Ensure time is datetime
    da["time"] = pd.to_datetime(da["time"].values)

    # Resample using the identified time dimension
    da_resampled = da.resample(time=scale).sum()
    return da_resampled


def plot_percent_volume_boxplot(df, output_path=None):
    """
    Create boxplots of percent positive volume by month, separated by GRP type.
    """
    plt.figure(figsize=(14, 8))

    # Define month order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Filter to only include months that are present in the data
    month_order = [m for m in month_order if m in df['month'].unique()]

    # Create boxplot using seaborn
    ax = sns.boxplot(x='month', y='percent_positive', hue='GRP_type', data=df, order=month_order)

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title('Percent Volume with Positive GRP Values by Month')
    plt.xlabel('Month')
    plt.ylabel('Percent Positive Volume (%)')
    plt.legend(title='GRP Type')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.close()


def plot_combined_barplot(df, group_by='year', hue='MU', output_path=None):
    """
    Create a bar plot of percent positive volume.
    """
    plt.figure(figsize=(14, 8))

    # Calculate mean values for each group
    summary_df = df.groupby([group_by, hue])['percent_positive'].mean().reset_index()

    # Sort if using month as group_by
    if group_by == 'month':
        # Define month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Filter to only include months that are present in the data
        month_order = [m for m in month_order if m in summary_df['month'].unique()]

        # Create bar plot with proper month ordering
        ax = sns.barplot(x=group_by, y='percent_positive', hue=hue, data=summary_df, order=month_order)
    else:
        # For other group_by values, no special ordering needed
        ax = sns.barplot(x=group_by, y='percent_positive', hue=hue, data=summary_df)

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title(f'Average Percent Volume with Positive GRP Values by {group_by}')
    plt.xlabel(group_by.capitalize())
    plt.ylabel('Percent Positive Volume (%)')
    plt.legend(title=hue)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.close()


def extract_parameters_from_filename(filename):
    """
    Extract mass and P coefficients from filename like "grp_m200_p0.35_*.nc"
    """
    mass_match = re.search(r'm(\d+)', filename)
    p_match = re.search(r'p([\d\.]+)', filename)

    mass = mass_match.group(1) if mass_match else "unknown"
    p_value = p_match.group(1) if p_match else "unknown"

    return mass, p_value


def extract_year(filename):
    """
    Extract a 4-digit year from a filename.
    """
    m = re.search(r'(\d{4})', filename)
    return m.group(1) if m else "Unknown"


# ---------------------------
# Main Processing Loop for Volume Time Series
# ---------------------------
# List to store all records
records = []

for file in processed_files:
    filename = os.path.basename(file)
    logger.info(f"Processing file: {filename}")

    # Extract parameters from filename
    mass, p_value = extract_parameters_from_filename(filename)
    year = extract_year(filename)

    # Create a parameter set identifier
    param_set = f"m{mass}_p{p_value}"

    try:
        ds = xr.open_dataset(file)

        # Ensure the time coordinate is in datetime format
        ds["time"] = pd.to_datetime(ds["time"].values)

        # Check if layer_volume exists
        if "layer_volume" not in ds:
            logger.error(f"layer_volume not found in {filename}, skipping")
            continue

        # Process both GRP and GRP_lethal if available
        grp_vars = []
        if "GRP" in ds:
            grp_vars.append("GRP")
        if "GRP_lethal" in ds:
            grp_vars.append("GRP_lethal")

        if not grp_vars:
            logger.warning(f"No GRP variables found in {filename}, skipping")
            continue

        for var_name in grp_vars:
            logger.info(f"Processing {var_name} for {param_set} in {year}")

            try:
                # First process the full lake
                vol_ts, total_vol_ts = compute_volume_time_series(
                    ds, var_name=var_name, threshold=grp_threshold
                )

                # Resample both time series
                vol_ts_resampled = resample_time_series(vol_ts, scale=resample_scale)
                total_vol_ts_resampled = resample_time_series(total_vol_ts, scale=resample_scale)

                # Calculate percent positive volume
                vol_percent_resampled = (vol_ts_resampled / total_vol_ts_resampled) * 100

                # Convert to pandas Series for easier handling
                vol_percent_series = vol_percent_resampled.to_series()

                # Debug log to see values
                logger.info(
                    f"Full lake {var_name} stats: min={vol_percent_series.min()}, max={vol_percent_series.max()}, mean={vol_percent_series.mean()}")

                # Add to records for the full lake
                for date, value in vol_percent_series.items():
                    month = date.strftime('%b')  # Abbreviated month name
                    month_num = date.month
                    records.append({
                        'year': year,
                        'month': month,
                        'month_num': month_num,
                        'date': date,
                        'MU': 'Full Lake',
                        'GRP_type': var_name,
                        'param_set': param_set,
                        'mass': mass,
                        'p_value': p_value,
                        'percent_positive': value
                    })

                # Now process each management unit
                for mu in [1, 2, 3, 4]:
                    mu_name = f"MU{mu}"
                    logger.info(f"Processing {var_name} for {param_set} in {year} for {mu_name}")

                    # Get indices for this MU
                    indices = np.where(mu_mask == mu)[0]

                    # Check if indices exist in the dataset's nele dimension
                    if len(indices) == 0:
                        logger.warning(f"No elements found for {mu_name}, skipping")
                        continue

                    # Filter indices that are within range
                    valid_indices = indices[indices < ds.sizes["nele"]]

                    if len(valid_indices) == 0:
                        logger.warning(f"No valid elements found for {mu_name} in {filename}, skipping")
                        continue

                    # Subset the dataset for this MU
                    ds_mu = ds.isel(nele=valid_indices)

                    # Compute volume time series for this MU
                    vol_ts_mu, total_vol_ts_mu = compute_volume_time_series(
                        ds_mu, var_name=var_name, threshold=grp_threshold
                    )

                    # Resample
                    vol_ts_mu_resampled = resample_time_series(vol_ts_mu, scale=resample_scale)
                    total_vol_ts_mu_resampled = resample_time_series(total_vol_ts_mu, scale=resample_scale)

                    # Calculate percent positive volume
                    vol_percent_mu_resampled = (vol_ts_mu_resampled / total_vol_ts_mu_resampled) * 100

                    # Convert to pandas Series
                    vol_percent_mu_series = vol_percent_mu_resampled.to_series()

                    # Debug log to see values
                    logger.info(
                        f"{mu_name} {var_name} stats: min={vol_percent_mu_series.min()}, max={vol_percent_mu_series.max()}, mean={vol_percent_mu_series.mean()}")

                    # Add to records for this MU
                    for date, value in vol_percent_mu_series.items():
                        month = date.strftime('%b')  # Abbreviated month name
                        month_num = date.month
                        records.append({
                            'year': year,
                            'month': month,
                            'month_num': month_num,
                            'date': date,
                            'MU': mu_name,
                            'GRP_type': var_name,
                            'param_set': param_set,
                            'mass': mass,
                            'p_value': p_value,
                            'percent_positive': value
                        })
            except Exception as e:
                logger.error(f"Error processing {var_name} in {filename}: {str(e)}")
                continue

        ds.close()

    except Exception as e:
        logger.error(f"Error opening {filename}: {str(e)}")
        continue

    gc.collect()  # Free memory

# ---------------------------
# Create DataFrame and Generate Plots
# ---------------------------
if not records:
    logger.warning("No records were created. Check for errors in processing.")
    exit(1)

df = pd.DataFrame(records)

# Add debugging info
logger.info(f"DataFrame created with {len(df)} records")
logger.info(
    f"Value statistics: min={df['percent_positive'].min()}, max={df['percent_positive'].max()}, mean={df['percent_positive'].mean()}")

# Save the complete table
output_table_path = os.path.join(PLOTS_DIR, "complete_GRP_volume_data.csv")
df.to_csv(output_table_path, index=False)
logger.info(f"Complete data table saved to {output_table_path}")

# Generate boxplots for each parameter set and MU
for param_set in df['param_set'].unique():
    # Filter for this parameter set
    df_param = df[df['param_set'] == param_set]

    # 1. Full Lake boxplot by month
    df_full = df_param[df_param['MU'] == 'Full Lake']
    if not df_full.empty:
        plot_percent_volume_boxplot(
            df_full,
            output_path=os.path.join(PLOTS_DIR, f"{param_set}_full_lake_monthly_boxplot.png")
        )

    # 2. Each MU boxplot by month
    for mu in df_param['MU'].unique():
        if mu == 'Full Lake':
            continue

        df_mu = df_param[df_param['MU'] == mu]
        if not df_mu.empty:
            plot_percent_volume_boxplot(
                df_mu,
                output_path=os.path.join(PLOTS_DIR, f"{param_set}_{mu}_monthly_boxplot.png")
            )

    # 3. Comparison across MUs (bar plot)
    plot_combined_barplot(
        df_param,
        group_by='month',
        hue='MU',
        output_path=os.path.join(PLOTS_DIR, f"{param_set}_MU_comparison_barplot.png")
    )

    # 4. Comparison of GRP vs GRP_lethal (bar plot)
    df_grp_compare = df_param[df_param['MU'] == 'Full Lake']
    if len(df_grp_compare['GRP_type'].unique()) > 1:
        plot_combined_barplot(
            df_grp_compare,
            group_by='month',
            hue='GRP_type',
            output_path=os.path.join(PLOTS_DIR, f"{param_set}_GRP_types_comparison_barplot.png")
        )

# Generate multiyear comparison for each MU
for mu in df['MU'].unique():
    df_mu_all = df[df['MU'] == mu]

    # Bar plot of years by GRP type
    plot_combined_barplot(
        df_mu_all,
        group_by='year',
        hue='GRP_type',
        output_path=os.path.join(PLOTS_DIR, f"{mu}_yearly_GRP_comparison_barplot.png")
    )

    # Bar plot of parameter sets
    plot_combined_barplot(
        df_mu_all,
        group_by='param_set',
        hue='GRP_type',
        output_path=os.path.join(PLOTS_DIR, f"{mu}_parameter_comparison_barplot.png")
    )

logger.info("Processing complete. All plots and tables have been saved.")