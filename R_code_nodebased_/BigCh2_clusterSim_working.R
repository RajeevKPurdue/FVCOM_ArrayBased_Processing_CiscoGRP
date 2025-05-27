# Load required packages
library(dplyr)
library(raster)
library(PBSmapping)
library(RColorBrewer)
library(ncdf4)
library(parallel)
library(akima)
library(reshape2)
library(ggplot2)
library(data.table)
library(abind)
library(sp)
library(rgdal)

# Set the main directory path (adjust this to your actual path)
main_dir <- "/Volumes/WD Backup/Rowe_FVCOM_data"

# Correctly set the data directory path
data_dir <- file.path(main_dir, "20170901_1_2013")  # This assumes your .nc files are in the "20170530" folder

# List all .nc files recursively within the directory
file_list <- list.files(data_dir, pattern = "\\.nc$", full.names = TRUE, recursive = TRUE)

# Print file_list to confirm it is populated
print(file_list)

# Correct the output directory path
output_dir <- file.path(main_dir, "Model_output_2013")
dir.create(output_dir, showWarnings = FALSE)  # Create the output directory if it doesn't exist

# Print output_dir to confirm it's correctly set
print(output_dir)

# Set working directory to the main folder containing your scripts and data
setwd(main_dir)

coord_dir <- file.path(main_dir, "FVCOM_coords")  # This assumes your .nc files are in the "20170530" folder
func_dir <- file.path(main_dir, "Scipts_Funcs")

# Load functions
source("/Volumes/WD Backup/Rowe_FVCOM_data/Scripts_Funcs/proj_functions.R")
source("/Volumes/WD Backup/Rowe_FVCOM_data/Scripts_Funcs/getColors.R")

# Load spatial data files
load(file.path(coord_dir, "erie05lcc_grid_polys.Rdata"))
load(file.path(coord_dir,"mask2.Rdata"))
load(file.path(coord_dir,"erie05lcc_grid_coords.Rdata"))

# Load basin coordinate subsets from the correct location
#western_basin_coords<- 
load(file.path(main_dir, "Rowe_et_al/western_basin_coords.Rdata"))
#central_basin_coords<- 
load(file.path(main_dir, "Rowe_et_al/central_basin_coords.Rdata"))
#eastern_basin_coords<- 
load(file.path(main_dir, "Rowe_et_al/eastern_basin_coords.Rdata"))


# Project the basin coordinates to match the projection system used in your analysis
#western_basin_coords <- projLcc(western_basin, prj_new)
#central_basin_coords <- projLcc(central_basin, prj_new)
#eastern_basin_coords <- projLcc(eastern_basin, prj_new)

# Project the main grid coordinates
#coordsn <- projLcc(coordsn, prj_new)

# Create indices based on matching coordinates
west_basin_indices <- which(coordsn$lon %in% western_basin$lon & coordsn$lat %in% western_basin$lat)
central_basin_indices <- which(coordsn$lon %in% central_basin$lon & coordsn$lat %in% central_basin$lat)
east_basin_indices <- which(coordsn$lon %in% eastern_basin$lon & coordsn$lat %in% eastern_basin$lat)

# Parameters
params <- list(
  CA = 1.61, CB = -0.538, CQ = 3.53, CTO = 16.8, CTM = 26.0, P = 0.4,
  RA = 0.0018, RB = -0.12, RQ = 0.047, RTO = 0.025, RTM = 0.0, RTL = 0.0,
  RK1 = 7.23, RK4 = 0.025, BACT = 0.0, SDA = 0.17, FA = 0.25, UA = 0.1,
  ED = 6500, OCC = 13556.0, AD = 2000
)

# GRP Function
GRP_func <- function(temp, DO, params) {
  DOcrit <- 0.138 * temp + 2.09
  fDO <- pmin(DO / DOcrit, 1.0)
  
  Vel <- params$RK1 * 200^(0.25)
  ACT <- exp(params$RTO * Vel)
  
  V <- (params$CTM - temp) / (params$CTM - params$CTO)
  V <- pmax(V, 0.0)
  Z <- log(params$CQ) * (params$CTM - params$CTO)
  Y <- log(params$CQ) * (params$CTM - params$CTO + 2.0)
  X <- (Z^2) * (1.0 + sqrt(1.0 + 40.0 / Y))^2 / 400.0
  Ft <- (V^X) * exp(X * (1.0 - V))
  
  Cmax <- params$CA * (200^params$CB)
  C <- Cmax * Ft * params$P * fDO
  Fu <- params$FA * C
  S <- params$SDA * (C - Fu)
  Ftr <- exp(params$RQ * temp)
  R <- params$RA * (200^params$RB) * Ftr * ACT
  U <- params$UA * (C - Fu)
  GRP <- C - (S - Fu - U) * (params$AD / params$ED) - (R * params$OCC) / params$ED
  
  return(GRP)
}

# Parallel processing setup
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
clusterExport(cl, c("GRP_func", "params"))

# Process each file individually
for (file_path in file_list) {
  
  print(paste("Processing file:", file_path))  # Print the file being processed
  
  # Process the file
  nc <- nc_open(file_path)
  
  # Extract lat, lon, and other variables
  lat <- ncvar_get(nc, "lat")
  lon <- ncvar_get(nc, "lon") - 360
  Times <- ncvar_get(nc, "Times", start = c(1, 1), count = c(-1, -1))
  time2 <- as.POSIXct(Times, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  daychar <- format(time2, "%Y-%m-%d")
  days <- unique(daychar)
  
  temp <- ncvar_get(nc, "temp", start = c(1, 1, 1), count = c(6106, 20, -1))
  do <- ncvar_get(nc, "Dissolved_oxygen", start = c(1, 1, 1), count = c(6106, 20, -1))
  zeta <- ncvar_get(nc, "zeta", start = c(1, 1), count = c(6106, -1))
  h <- ncvar_get(nc, "h", start = c(1), count = c(6106))
  siglay <- ncvar_get(nc, "siglay", start = c(1, 1), count = c(-1, -1))[1,]
  siglev <- ncvar_get(nc, "siglev", start = c(1, 1), count = c(-1, -1))[1,]
  
  nc_close(nc)
  
  # Export necessary variables to the worker nodes
  clusterExport(cl, c("daychar", "days", "temp", "do", "zeta", "h", "siglay", "siglev", "Times", "time2", "lat", "lon"))
  
  print("Running GRP function...")  # Print before running GRP function
  
  # Process the GRP function for each time slice
  daily_means <- parLapply(cl, unique(daychar), function(day1) {
    xx <- which(daychar == day1)
    temp_mean <- apply(temp[,,xx], MARGIN = c(1, 2), mean)
    do_mean <- apply(do[,,xx], MARGIN = c(1, 2), mean)
    GRP <- GRP_func(temp_mean, do_mean, params)
    return(list(GRP = GRP, temp = temp_mean, DO = do_mean))
  })
  
  print("Combining results...")  # Print after running GRP function
  
  # Combine the results
  chunk_result <- lapply(c("GRP", "temp", "DO"), function(var) {
    do.call(abind, c(lapply(daily_means, `[[`, var), list(along = 3)))
  })
  
  names(chunk_result) <- c("GRP", "temp", "DO")
  
  # Save the results for this chunk (file)
  file_base <- tools::file_path_sans_ext(basename(file_path))
  
  saveRDS(chunk_result$GRP, file.path(output_dir, paste0(file_base, "_GRP.rds")))
  saveRDS(chunk_result$temp, file.path(output_dir, paste0(file_base, "_temp.rds")))
  saveRDS(chunk_result$DO, file.path(output_dir, paste0(file_base, "_DO.rds")))
  
  # Subset GRP by basin and save
  west_basin_GRP <- chunk_result$GRP[west_basin_indices, , , drop = FALSE]
  central_basin_GRP <- chunk_result$GRP[central_basin_indices, , , drop = FALSE]
  east_basin_GRP <- chunk_result$GRP[east_basin_indices, , , drop = FALSE]
  
  saveRDS(west_basin_GRP, file.path(output_dir, paste0(file_base, "_west_GRP.rds")))
  saveRDS(central_basin_GRP, file.path(output_dir, paste0(file_base, "_central_GRP.rds")))
  saveRDS(east_basin_GRP, file.path(output_dir, paste0(file_base, "_east_GRP.rds")))
  
  # Repeat for temp and DO as needed...
  west_basin_temp <- chunk_result$temp[west_basin_indices, , , drop = FALSE]
  central_basin_temp <- chunk_result$temp[central_basin_indices, , , drop = FALSE]
  east_basin_temp <- chunk_result$temp[east_basin_indices, , , drop = FALSE]
  
  saveRDS(west_basin_temp, file.path(output_dir, paste0(file_base, "_west_temp.rds")))
  saveRDS(central_basin_temp, file.path(output_dir, paste0(file_base, "_central_temp.rds")))
  saveRDS(east_basin_temp, file.path(output_dir, paste0(file_base, "_east_temp.rds")))
  
  west_basin_DO <- chunk_result$DO[west_basin_indices, , , drop = FALSE]
  central_basin_DO <- chunk_result$DO[central_basin_indices, , , drop = FALSE]
  east_basin_DO <- chunk_result$DO[east_basin_indices, , , drop = FALSE]
  
  saveRDS(west_basin_DO, file.path(output_dir, paste0(file_base, "_west_DO.rds")))
  saveRDS(central_basin_DO, file.path(output_dir, paste0(file_base, "_central_DO.rds")))
  saveRDS(east_basin_DO, file.path(output_dir, paste0(file_base, "_east_DO.rds")))
  
  # Run garbage collection after processing each file to free up memory
  rm(temp, do, chunk_result, west_basin_GRP, central_basin_GRP, east_basin_GRP)
  gc()
}

# Stop parallel cluster
stopCluster(cl)

