#!/usr/bin/env python3
import os
import gc
import logging
import numpy as np
import xarray as xr
import pandas as pd

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Growth Model Function (unchanged)
# ---------------------------
def calculate_growth_fdos(DO_array, Temp_array, mass_coeff, P_coeff, AD_arr, slope_val, intercept_val):
    """
    Calculate Growth Rate Potential (GRP) and GRP_lethal using volumetric
    dissolved oxygen and temperature arrays (assumed 3D: time, nele, num_layers)
    along with scalar parameters.
    """
    # 1. Calculate the critical dissolved oxygen (DOcrit) as a linear function of temperature.
    DOcrit = slope_val * Temp_array + intercept_val

    # 2. Compute the normalized DO fraction (fDO) and cap values to 1.
    fDO = np.minimum(DO_array / DOcrit, 1.0)

    # 3. Create a uniform mass array from mass_coeff.
    mass = np.full_like(Temp_array, mass_coeff)

    # 4. Prepare a temperature variable for lethal calculations.
    t_lethal = Temp_array

    # 5. Calculate a lethal DO threshold (DO_lethal) as an exponential function of temperature.
    DO_lethal = 0.4 + 0.000006 * np.exp(0.59 * t_lethal)

    # 6. Calculate the lethal DO fraction (fdo_lethal) and clip between 0 and 1.
    fdo_lethal = np.clip((DO_array - DO_lethal) / DOcrit, 0, 1)

    # 7. Clip fDO to ensure it remains between 0 and 1.
    fDO = np.clip(fDO, 0, 1)

    # 8. Define consumption parameters.
    CA, CB, CQ = 1.61, -0.538, 3.53
    CTO, CTM = 16.8, 26.0

    # 9. Define respiration parameters.
    RA, RB, RQ = 0.0018, -0.12, 0.047
    RTO, RK1 = 0.025, 7.23

    # 10. Compute velocity (Vel) and the activity factor (ACT).
    Vel = RK1 * np.power(mass, 0.025)
    ACT = np.exp(RTO * Vel)

    # 11. Create an AD array and set a benthic (bottom layer) value.
    AD = np.full_like(Temp_array, AD_arr, dtype=np.float64)
    AD[:, :, -1] = 3138.0  # Use literature value for benthic layer - Schaeffer et al.

    # 12. Scale consumption by DO fractions.
    P = P_coeff * fDO
    P_lethal = P_coeff * fdo_lethal

    # 13. Compute a temperature-dependent modifier for consumption.
    V = (CTM - Temp_array) / (CTM - CTO)
    V = np.maximum(V, 0.0)
    Z = np.log(CQ) * (CTM - CTO)
    Y = np.log(CQ) * (CTM - CTO + 2.0)
    X = (Z**2) * (1.0 + ((1.0 + 40.0) / Y)**0.5)**2 / 400.0
    Ft = (V**X) * np.exp(X * (1.0 - V))

    # 14. Calculate the maximum consumption rate.
    Cmax = CA * np.power(mass, CB)
    # 15. Calculate actual consumption and lethal consumption.
    C = Cmax * Ft * P
    C_lethal = Cmax * Ft * P_lethal

    # 16. Define egestion and excretion factors.
    FA, SDA, UA = 0.25, 0.17, 0.1
    F = FA * C
    S = SDA * (C - F)

    # 17. Compute respiration factor.
    Ftr = np.exp(RQ * Temp_array)
    R = RA * np.power(mass, RB) * Ftr * ACT

    # 18. Calculate excretion.
    U = UA * (C - F)

    # 19. Define energy density and scaling constant.
    ED, OCC = 6500.0, 13556.0

    # 20. Compute Growth Rate Potential (GRP) and a lethal variant.
    GRP = C - (S + F + U) * (AD / ED) - (R * OCC) / ED
    GRP_lethal = C_lethal - (S + F + U) * (AD / ED) - (R * OCC) / ED

    return GRP, GRP_lethal

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":

    # Define parameter arrays (2 mass coefficients x 3 P coefficients = 6 combinations)
    mass_coefficients = [200, 500]
    P_coefficients = [0.35, 0.5, 0.65]
    AD_arr = 2512          # Example AD value
    slope_val = 0.168       # Example slope
    intercept_val = 1.63    # Example intercept

    # Define directory paths for processed (daily aggregated) summary files and output directory
    PROCESSED_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2'  # e.g., spatial_summary_2017.nc, etc.
    OUTPUT_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_volumes_Rnd2_AD2000/reiterated_AD_2500'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get list of processed files (assuming one file per year)
    processed_files = [
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.startswith("spatial_summary_") and f.endswith(".nc")
    ]
    processed_files.sort()

    # Loop over each processed file
    for file in processed_files:
        logger.info(f"Processing file: {file}")

        # Use a context manager to automatically close the file when done
        with xr.open_dataset(file) as ds:
            # Ensure the time coordinate is a proper datetime index
            ds["time"] = pd.to_datetime(ds["time"].values)

            # Loop over each parameter combination
            for m in mass_coefficients:
                for p in P_coefficients:
                    logger.info(f"  Running model for mass={m}, P={p}")

                    # Get the spatiotemporal DO and temperature arrays (assumed 3D: time, nele, num_layers)
                    DO_array = ds["dissolved_oxygen"].values
                    Temp_array = ds["temperature"].values
                    depth_array = ds["depth"].values  # Newly added extraction

                    # Compute growth potential using the given parameters
                    GRP, GRP_lethal = calculate_growth_fdos(
                        DO_array=DO_array,
                        Temp_array=Temp_array,
                        mass_coeff=m,
                        P_coeff=p,
                        AD_arr=AD_arr,
                        slope_val=slope_val,
                        intercept_val=intercept_val
                    )

                    # Create a new Dataset containing GRP and GRP_lethal
                    ds_growth = xr.Dataset(
                        {
                            "GRP": (("time", "nele", "num_layers"), GRP),
                            "GRP_lethal": (("time", "nele", "num_layers"), GRP_lethal),
                            "depth": (("time", "nele", "num_layers"), depth_array)
                        },
                        coords=ds.coords  # Retain the original coordinates
                    )
                    # Optionally, add the parameter values as attributes
                    ds_growth = ds_growth.assign_attrs(mass_coeff=m, P_coeff=p)

                    # Construct an output filename that includes the base name and parameter info
                    base_name = os.path.basename(file).replace(".nc", "")
                    output_file = os.path.join(
                        OUTPUT_DIR,
                        f"{base_name}_growth_mass{m}_P{p}.nc"
                    )
                    logger.info(f"    Writing output to: {output_file}")

                    # Write the growth dataset to disk
                    ds_growth.to_netcdf(output_file)
                    ds_growth.close()

        # Force garbage collection to free memory before processing the next file
        gc.collect()
        logger.info(f"Completed processing file: {file}")
#%%

"trying to truly assign chironomid to benthos and not last siglay"
import os
import gc
import numpy as np
import xarray as xr
import pandas as pd
import logging

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_growth_fdos(DO_array, Temp_array, depth_array, mass_coeff, P_coeff, AD_arr, slope_val, intercept_val):
    """
    Calculate Growth Rate Potential (GRP) and GRP_lethal using volumetric dissolved oxygen,
    temperature, and depth arrays (3D: time, nele, num_layers) along with scalar parameters.
    """
    # 1. Compute the critical DO as a linear function of temperature.
    DOcrit = slope_val * Temp_array + intercept_val

    # 2. Compute normalized DO fraction (fDO), capped at 1.
    fDO = np.minimum(DO_array / DOcrit, 1.0)
    fDO = np.clip(fDO, 0, 1)

    # 3. Create a uniform mass array from mass_coeff.
    mass = np.full_like(Temp_array, mass_coeff, dtype=np.float64)

    # 4. Prepare a temperature variable for lethal calculations.
    t_lethal = Temp_array

    # 5. Compute a lethal DO threshold as an exponential function of temperature.
    DO_lethal = 0.4 + 0.000006 * np.exp(0.59 * t_lethal)

    # 6. Compute lethal DO fraction (fdo_lethal) and clip between 0 and 1.
    fdo_lethal = np.clip((DO_array - DO_lethal) / DOcrit, 0, 1)

    # 7. Create AD array with default value AD_arr
    AD = np.full_like(Temp_array, AD_arr, dtype=np.float64)

    # 8. Modify AD for the benthic (bottom) portion only.
    bottom_layer_depth = depth_array[:, :, -1]       # shape: (time, nele)
    max_depth_per_cell = np.max(depth_array, axis=2)   # maximum depth for each (time, nele)
    tol = 0.1  # tolerance in the same units as depth (adjust as needed)
    mask_benthic = np.isclose(bottom_layer_depth, max_depth_per_cell, atol=tol)

    # Apply the benthic AD value (3138.0) only where the mask is True
    AD_bottom = AD[:, :, -1]
    AD_bottom[mask_benthic] = 3138.0
    AD[:, :, -1] = AD_bottom  # update the bottom layer in AD

    # Define consumption parameters.
    CA, CB, CQ = 1.61, -0.538, 3.53
    CTO, CTM = 16.8, 26.0

    # Define respiration parameters.
    RA, RB, RQ = 0.0018, -0.12, 0.047
    RTO, RK1 = 0.025, 7.23

    Vel = RK1 * np.power(mass, 0.025)
    ACT = np.exp(RTO * Vel)

    # Scale consumption by DO fractions.
    P = P_coeff * fDO
    P_lethal = P_coeff * fdo_lethal

    # Compute temperature-dependent modifier for consumption.
    V = (CTM - Temp_array) / (CTM - CTO)
    V = np.maximum(V, 0.0)
    Z = np.log(CQ) * (CTM - CTO)
    Y = np.log(CQ) * (CTM - CTO + 2.0)
    X = (Z**2) * (1.0 + ((1.0 + 40.0) / Y)**0.5)**2 / 400.0
    Ft = (V**X) * np.exp(X * (1.0 - V))

    # Calculate maximum consumption rate.
    Cmax = CA * np.power(mass, CB)
    C = Cmax * Ft * P
    C_lethal = Cmax * Ft * P_lethal

    # Define egestion and excretion factors.
    FA, SDA, UA = 0.25, 0.17, 0.1
    F = FA * C
    S = SDA * (C - F)

    # Compute respiration.
    Ftr = np.exp(RQ * Temp_array)
    R = RA * np.power(mass, RB) * Ftr * ACT

    # Calculate excretion.
    U = UA * (C - F)

    # Define energy density and scaling constant.
    ED, OCC = 6500.0, 13556.0

    # Compute GRP and GRP_lethal.
    GRP = C - (S + F + U) * (AD / ED) - (R * OCC) / ED
    GRP_lethal = C_lethal - (S + F + U) * (AD / ED) - (R * OCC) / ED

    return GRP, GRP_lethal

# ---------------------------------------------------------
# Main script: Loading processed files and running GRP model
# ---------------------------------------------------------
if __name__ == "__main__":
    # Define parameter arrays (e.g., 2 mass coefficients x 3 P coefficients = 6 combinations)
    mass_coefficients = [200, 500]
    P_coefficients = [0.35, 0.5, 0.65]
    AD_arr = 2512          # Default AD value for non-benthic portions
    slope_val = 0.168       # Example slope for DOcrit calculation
    intercept_val = 1.63    # Example intercept for DOcrit calculation

    # Directory paths for processed (daily aggregated) summary files and GRP outputs
    PROCESSED_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/Outputs_Volumes_rnd2'
    OUTPUT_DIR = '/Volumes/WD Backup/Rowe_FVCOM_data/GRP_volumes_Rnd2_AD2000/reiterated_AD_2500'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get list of processed files (assuming one file per year)
    processed_files = [
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.startswith("spatial_summary_") and f.endswith(".nc")
    ]
    processed_files.sort()

    for file in processed_files:
        logger.info(f"Processing file: {file}")
        with xr.open_dataset(file) as ds:
            # Ensure the time coordinate is a proper datetime index
            ds["time"] = pd.to_datetime(ds["time"].values)

            # Extract the 3D arrays: dissolved oxygen, temperature, and depth.
            DO_array = ds["dissolved_oxygen"].values
            Temp_array = ds["temperature"].values
            depth_array = ds["depth"].values

            # Loop over each parameter combination.
            for m in mass_coefficients:
                for p in P_coefficients:
                    logger.info(f"  Running model for mass={m}, P={p}")
                    GRP, GRP_lethal = calculate_growth_fdos(
                        DO_array=DO_array,
                        Temp_array=Temp_array,
                        depth_array=depth_array,
                        mass_coeff=m,
                        P_coeff=p,
                        AD_arr=AD_arr,
                        slope_val=slope_val,
                        intercept_val=intercept_val
                    )

                    # Create a new Dataset containing GRP, GRP_lethal, and depth.
                    ds_growth = xr.Dataset(
                        {
                            "GRP": (("time", "nele", "num_layers"), GRP),
                            "GRP_lethal": (("time", "nele", "num_layers"), GRP_lethal),
                            "depth": (("time", "nele", "num_layers"), depth_array)
                        },
                        coords=ds.coords  # Retain the original coordinates
                    )
                    # Optionally add the parameter values as attributes.
                    ds_growth = ds_growth.assign_attrs(mass_coeff=m, P_coeff=p)

                    # Define the output file name.
                    base_name = os.path.basename(file)
                    output_file = os.path.join(OUTPUT_DIR, f"grp_m{m}_p{p}_" + base_name)
                    ds_growth.to_netcdf(output_file)
                    ds_growth.close()
                    logger.info(f"  Written output to {output_file}")

    # Force garbage collection after processing all files.
    gc.collect()
