library(dplyr)      # Data manipulation
library(reshape2)   # Data reshaping
library(ggplot2)    # Plotting
library(parallel)   # Parallel processing
library(PBSmapping) # Spatial plotting (plotPolys)
library(gifski)     # GIF creation
library(animation)  # GIF creation and handling

######## BEFORE RUNNING ###########

# Change/Check main_dir, data_dir, Output_dir, animation dir
# Check the length of files & the daychar per file - ESP LAST FILE
# Change the output file for the .gif

############################ CHECKLIST COMPLETE? #######

# Set working directory and define directories
main_dir <- "/Volumes/WD Backup/Rowe_FVCOM_data"

data_dir <- file.path(main_dir, "Model_output_2015")

frame_dir <- "animation_frames_2015"

# 2015 = 8 files - first days = (03-31 to 04-29); last days = (10-27 to 11-22) = len(27 for daychar)

# Ensure the directory exists
dir.create(frame_dir, showWarnings = FALSE)

# Define the start date
start_date <- as.Date("2015-03-31")

# Initialize an empty list to store data frames from each file
df_combined_all_whole <- list()

# Parallel processing setup
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
clusterExport(cl, c("start_date", "melt", "as.Date", "seq"))
# List all the GRP files for the whole lake
files <- list.files(data_dir, pattern = paste0("erie_\\d{4}_GRP\\.rds$"), full.names = TRUE)
print(files)
# Track the current start date
current_date <- start_date

# Loop over files
for (file_idx in seq_along(files)) {
  # Load the GRP data
  grp_data <- readRDS(files[file_idx])
  
  # Calculate the date range for the current file
  num_days <- if (file_idx < 8) 30 else 27  # 30 days for files 1-7, 7 days for file 8 # 1:(n-1) (30), n = len(daychar) 
  date_seq <- seq(current_date, by = "day", length.out = num_days)
  
  # Convert to long format
  df_long <- melt(grp_data)
  colnames(df_long) <- c("Node", "Depth", "Time", "GRP")
  
  # Assign the correct Date values based on the sequence
  df_long$Date <- date_seq[df_long$Time]
  
  # Append to list
  df_combined_all_whole[[length(df_combined_all_whole) + 1]] <- df_long
  
  # Update the current_date for the next file
  current_date <- current_date + num_days
}

# Combine all data into a single data frame
df_combined_all <- bind_rows(df_combined_all_whole)


print(unique(df_combined_all$Date))

## Remove objects to free up memory
rm(files, df_combined_all_whole, grp_data, date_seq, num_days, df_long, current_date, file_idx)

# Explicitly run garbage collection to free up memory
gc()




# Create Depth_Group variable
df_combined_all$Depth_Group <- cut(       
  df_combined_all$Depth,
  breaks = c(0, 5, 10, 15, 20),
  labels = c("1-5", "6-10", "11-15", "16-20"),
  include.lowest = TRUE
)


# Calculate mean GRP by Node, Date, and Depth_Group
df_summary <- df_combined_all %>%
  group_by(Node, Date, Depth_Group) %>%
  summarize(Mean_GRP = mean(GRP, na.rm = TRUE), .groups = "drop")

print(unique(df_combined_all$Depth_Group))
print(unique(df_summary$Depth_Group))
print(head(df_summary))
print(unique(df_summary$Date))
print(length(unique(df_summary$Node)))

## Working on Spatial Projection and graphics
source("proj_functions.R") # Load Mark's functions 
source("GetColors_GRP.R") # modfied Mark's for GRP brewer Zissuo + 'getColorsVal = 20 breaks w viridis'

# Load spatial data files
load("erie05lcc_grid_polys.Rdata")
load("mask2.Rdata")
load("erie05lcc_grid_coords.Rdata")


#  example 4x4
n <- layout(matrix(c(1,2,3,4), nrow = 2, ncol = 2, byrow = TRUE)
            ,widths=c(1,0.2),heights=c(1,1))
layout.show(n)

xlim = c(-5, 328) #xlimLcc #  c(min(xx$x),max(xx$x))
ylim = c(-1.06822, 153) # ylimLcc # c(min(xx$y),max(xx$y))
plotPolys(shoreline, col="white", projection = TRUE
          ,xlim=xlim
          ,ylim=ylim
          ,xlab="",ylab=""
          ,xaxt="n",yaxt="n"
)

# for this method, set the size of hte points so they just overlap a little
points(coordsn$X/1000, coordsn$Y/1000, pch=15, col=cols$cols, cex=1.4)
# plot again to restore some of the detail in finer resolution areas
points(coordsn$X/1000, coordsn$Y/1000, pch=15, col=cols$cols, cex=1.0)

# plot the land mask to cover areas outside the model domain
addPolys(mask2, col="tan")


# Choose a single date for testing
#test_date <- as.Date("2016-03-30")

# Subset data for the test date
#test_data <- df_summary %>% filter(Date == test_date)


# Set original xlim and ylim values
xlim <- c(-5, 325)  # Original xlim for the plot
ylim <- c(-1.06822, 150)  # Original ylim for the plot


################################## GIF code blocks - working ###########

# Create a directory to save frames
#frame_dir <- "animation_frames"
#dir.create(frame_dir, showWarnings = FALSE)



############## Trying new gif function with fixed colorbar vals


### Creating subset of test frames to quickly assess this block
unique_dates = unique(df_summary$Date)
test_dates <- unique_dates[1:5]

# Calculate global colorbar limits from the entire dataset
values_all <- df_summary$Mean_GRP
min_value <- min(df_summary$Mean_GRP, na.rm = TRUE)
min_val<- min_value - 0.01
max_value <- max(df_summary$Mean_GRP, na.rm = TRUE)
max_val <- max_val +0.01
interval <- 0.005  # Define your desired interval

# Use the actual min and max values to create the breaks
global_brks <- seq(min_val, max_val, by = interval)
global_cols <- getColorsVal(values = values_all, brk = global_brks, num_breaks = nlevels(global_brks))  # Get colors based on global breaks

# Correct layout with 6 panels: 4 for depth groups, 2 for colorbars
layout_matrix <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
#show(layout_matrix)


# Set up a GIF output
saveGIF({
  
  # Loop over each unique date to generate frames
  for (date in unique_dates) {                     #unique_dates
    test_data <- df_summary %>% filter(Date == date)
    
    # Initialize a new plot
    plot.new()
    
    # Set up a layout for the plots with two shared colorbars
    layout(layout_matrix, widths = c(1, 1, 0.25), heights = c(1, 1))
    
    # Define the panels where the depth groups will be placed
    panel_indices <- c(1, 2, 4, 5)
    
    # Plot for each of the first four panels (depth groups)
    for (i in seq_along(panel_indices)) {
      # Set the panel to plot in
      par(mfg = c(floor((panel_indices[i] - 1) / 3) + 1, (panel_indices[i] - 1) %% 3 + 1))
      
      depth_group <- unique(test_data$Depth_Group)[i]
      depth_data <- test_data %>% filter(Depth_Group == depth_group)
      
      # Calculate the GRP values for coloring
      values <- depth_data$Mean_GRP
      brks <- global_brks  # Use global breaks
      
      # Assuming getColorsVal is a valid function that returns a list with 'cols' and 'zlim'
      cols <- getColorsVal(values = values, brk = brks)  # Get colors based on breaks
      
      # Plotting using base R functions
      par(mar = c(4, 0, 5, 0))  # Negative left margin  # Adjust margins to shift plot to the left
      plotPolys(shoreline, col = "white", projection = TRUE,
                xlim = xlim, ylim = ylim,
                xlab = "", ylab = "", xaxt = "n", yaxt = "n")
      
      # Add the points with the appropriate color and size
      points(coordsn$X / 1000, coordsn$Y / 1000, pch = 15, col = cols$cols, cex = 1.405 ) #1.4 original for scale (1K)
      points(coordsn$X / 1000, coordsn$Y / 1000, pch = 15, col = cols$cols, cex = 1.05 )  #1.0 original for scale (800)
      
      # Plot the land mask to cover areas outside the model domain
      addPolys(mask2, col = "tan")
      
      # Add a title to the plot
      title(main = paste("Mean GRP on", format(as.Date(date), "%Y-%m-%d"), "- Depth Group (sigma layer):", depth_group), cex.main = 2)
    }
    
    
    # Plot the colorbars in panels 3 and 6
    for (panel in c(3, 6)) {
      # Set the panel to plot in
      par(mfg = c(floor((panel - 1) / 3) + 1, (panel - 1) %% 3 + 1))
      
      par(mar = c(5, 0, 5, 9.5))  # Adjust margins for colorbars
      lab.breaks <- sprintf("%1.3f", global_brks)  # Format the breaks for labels
      
      # Set up the z matrix for the colorbar
      z <- matrix(seq(min(global_cols$zlim), max(global_cols$zlim), length.out = length(global_cols$colorbar)), nrow = 1)
      
      # Plot the color scale
      image(seq(0.01, 0.02, length.out = 2), seq_along(global_cols$colorbar), z, col = global_cols$colorbar, axes = FALSE, xlab = "", ylab = "")
      axis(4, at = seq_along(lab.breaks), labels = lab.breaks, las = 2, cex.axis = 1.9, font = 2, tick = FALSE)
      
      # Add a label to the colorbar
      label <- expression(paste("GRP ", g, g^-1, d^-1))
      mtext(label, side = 3, line = 1, cex = 1.2)
    }
    
  }
  
}, movie.name = "GRP_Animation_2015_viridis.gif", interval = 0.25, ani.width = 1250, ani.height = 900, autobrowse = FALSE)

# To remove all objects before going to next basin block of code paritioning 

rm(rm(list = ls()))
############# Stop Cluster here #################
# Stop cluster 
stopCluster(cl)




########################### Second block - splitting by basins - ggplot comparisons ####

# Reload required packages 

library(dplyr)      # Data manipulation
library(reshape2)   # Data reshaping
library(ggplot2)    # Plotting
library(parallel)   # Parallel processing
library(PBSmapping) # Spatial plotting (plotPolys)
library(gifski)     # GIF creation
library(animation)  # GIF creation and handling

######## BEFORE RUNNING ###########

# Change/Check main_dir, data_dir, Output_dir, animation dir
# Check the length of files & the daychar per file - ESP LAST FILE
# Change the output file for the .gif

############################ CHECKLIST COMPLETE? #######

setwd("/Volumes/WD Backup/Rowe_FVCOM_data")

### Checking coords + nodes on the individual basin .RDS dat files 

load('western_basin_coords.Rdata')
load("/Volumes/WD Backup/Rowe_FVCOM_data/central_basin_coords.Rdata")
load("/Volumes/WD Backup/Rowe_FVCOM_data/eastern_basin_coords.Rdata")

#inspecting these data
#load("/Volumes/WD Backup/Rowe_FVCOM_data/chrp2017.Rdata") # These are the buoy data from 2017 

print(length(unique(western_basin$node)))
print(length(unique(central_basin$node)))
print(length(unique(eastern_basin$node)))

print(range(unique(western_basin$node))) # 52 2662
print(range(unique(central_basin$node))) # 1274 5696
print(range(unique(eastern_basin$node))) # 4282 6106

# Load functions
source("proj_functions.R")
source("getColors.R")

# Load spatial data files
load("erie05lcc_grid_polys.Rdata")
load("mask2.Rdata")
load("erie05lcc_grid_coords.Rdata")

library(ncdf4)

fn <- "/Volumes/WD Backup/Rowe_FVCOM_data/20170905_2/erie_0001.nc"
nc <- nc_open(fn, readunlim = FALSE)

lat <- ncvar_get(nc, "lat")
lon <- ncvar_get(nc, "lon") - 360
Times <- ncvar_get(nc, "Times", start = c(1, 1), count = c(-1, -1))
time2 <- as.POSIXct(Times, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
daychar <- format(time2, "%Y-%m-%d")
print(daychar)

# Projecting with nodes 1-52

print(min(shoreline$X))
print(min(shorelineLL$Y))
print(max(shorelineLL$X))
print(max(shorelineLL$Y))

xlim = c(-83.47519,  -78.854)
ylim = c(41.38285, 42.90551)

plotPolys(shorelineLL, 
          xlim=c(-83.47519,  -78.854),
          ylim=c(41.38285, 42.90551))

points(coordsn$lon, coordsn$lat, pch=".")
text(coordsn$lon, coordsn$lat, coordsn$node, cex=0.7)
points(western_basin$lon, western_basin$lat, col="white")
text(western_basin$lon, western_basin$lat, col="red", cex=0.7)
points(central_basin$lon, central_basin$lat, col="white")
text(central_basin$lon, central_basin$lat, col="green", cex=0.7)
points(eastern_basin$lon, eastern_basin$lat, col="white")
text(eastern_basin$lon, eastern_basin$lat, col="blue", cex=0.7)


plotPoints(shorelineLL, col = "white", projection = "llCRS"
           ,xlim=xlim
           ,ylim=ylim
           ,xlab="",ylab=""
           ,xaxt="n",yaxt="n"
)

#xlim = c(-5, 325) # xlimLcc # c(min(xx$x),max(xx$x))
#ylim = c(-1.06822, 150) # ylimLcc # c(min(xx$y),max(xx$y))
plotPolys(shorelineLL, col="white", projection = TRUE
          ,xlim=xlim
          ,ylim=ylim
          ,xlab="",ylab=""
          ,xaxt="n",yaxt="n"
)



# Create dirs + # Initialize the objects for export



# 2015 = 8 files - first days = (03-31 to 04-29); last days = (10-27 to 11-22) = len(27 for daychar)

# Ensure the directory exists
dir.create(frame_dir, showWarnings = FALSE)

# Define the start date
start_date <- as.Date("2015-03-31")

# Initialize an empty list to store data frames from each file
df_combined_all <- list()


# Parallel processing setup
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
clusterExport(cl, c("start_date", "melt", "as.Date", "seq"))


# Loop over basins
basins <- c("west", "central", "east")
for (basin in basins) {
  df_list <- list()  # List to store data for each basin
  
  # Load the GRP data for each basin
  basin_files <- list.files(data_dir, pattern = paste0("erie_\\d{4}_", basin, "_GRP\\.rds$"), full.names = TRUE)
  
  # Track the current start date
  current_date <- start_date
  
  for (file_idx in seq_along(basin_files)) {
    grp_data <- readRDS(basin_files[file_idx])
    
    # Calculate the date range for the current file
    num_days <- if (file_idx < 8) 30 else 27  # 30 days for files 1-7, 7 days for file 8
    date_seq <- seq(current_date, by = "day", length.out = num_days)
    
    # Convert to long format
    df_long <- melt(grp_data)
    colnames(df_long) <- c("Node", "Depth", "Time", "GRP")
    
    # Assign the correct Date values based on the sequence
    df_long$Date <- date_seq[df_long$Time]
    
    # Add a Basin column
    df_long$Basin <- basin
    
    # Append to list
    df_list[[length(df_list) + 1]] <- df_long
    
    # Update the current_date for the next file
    current_date <- current_date + num_days
  }
  
  df_combined <- bind_rows(df_list)
  df_combined_all[[basin]] <- df_combined  # Store combined data for this basin
}

df_combined_all <- bind_rows(df_combined_all)  # Combine data across all basins

# check if data has everything 

head(df_combined_all)
print(length(unique(df_combined_all$Node)))
print(length(unique(df_combined_all$Basin)))
print(length(unique(df_combined_all$Depth)))
print(length(unique(df_combined_all$Date)))
print(length(unique(df_combined_all$GRP)))







# Memory management: Clear intermediate objects
rm(df_list, df_combined, df_long, basin_files)
gc()  # Run garbage collection to free up memory







# Create depth group variable
df_combined_all$Depth_Group <- cut(
  df_combined_all$Depth,
  breaks = c(0, 5, 10, 15, 20),
  labels = c("1-5", "6-10", "11-15", "16-20"),
  include.lowest = TRUE
)


# Calculate mean GRP by Node, Date, and Depth_Group
#df_summary <- df_combined_all %>%
#  group_by(Node, Date, Depth_Group) %>%
#  summarize(Mean_GRP = mean(GRP, na.rm = TRUE), .groups = "drop")


# Split data by basin
df_west <- df_combined_all %>% filter(Basin == "west") %>%
  mutate(Date = as.Date(Date)) 

df_central <- df_combined_all %>% filter(Basin == "central") %>%
  mutate(Date = as.Date(Date)) 

df_east <- df_combined_all %>% filter(Basin == "east") %>%
  mutate(Date = as.Date(Date)) 

# Summarize GRP for the west basin
df_west_summary <- df_west %>%
  group_by(Node, Date, Depth_Group) %>%
  summarize(Mean_GRP = mean(GRP, na.rm = TRUE), .groups = "drop")

# Summarize GRP for the central basin
df_central_summary <- df_central %>%
  group_by(Node, Date, Depth_Group) %>%
  summarize(Mean_GRP = mean(GRP, na.rm = TRUE), .groups = "drop")

# Summarize GRP for the east basin
df_east_summary <- df_east %>%
  group_by(Node, Date, Depth_Group) %>%
  summarize(Mean_GRP = mean(GRP, na.rm = TRUE), .groups = "drop")


# Define a function to create and save plots
# Define a function to create and save plots with smoothing
plot_by_basin_with_smoothing <- function(df, basin_name) {
  p <- ggplot(df, aes(x = Date, y = GRP, color = Depth_Group)) +
    geom_line(stat = "summary", fun.data = "mean_cl_boot") +
    facet_wrap(~Depth_Group, scales = "free_y") +
    scale_x_date(date_breaks = "15 days", date_labels = "%m/%d/%Y") +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black"),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.margin = margin(t = 10, r = 10, b = 40, l = 10)
    ) +
    labs(title = paste("Smoothed GRP Time Series for", basin_name, "Basin"))
  
  # Save the plot
  ggsave(filename = paste0("Smoothed_GRP_Time_Series_", basin_name, "_Basin.png"), plot = p, width = 12, height = 8)
}

# Example usage:
# Assuming df_west_summary, df_central_summary, and df_east_summary are your data frames
plot_by_basin_with_smoothing(df_west_summary, "West")
plot_by_basin_with_smoothing(df_central_summary, "Central")
plot_by_basin_with_smoothing(df_east_summary, "East")


# Memory management
rm(df_combined_all, df_west, df_central, df_east)
gc()

# stop cluster when done

stopCluster(cl)





# Function to plot Mean_GRP for a single node
plot_node_grps <- function(df, node_id) {
  # Filter data for the specific node
  node_data <- df %>% filter(Node == node_id)
  
  # Create the plot
  p <- ggplot(node_data, aes(x = Date, y = Mean_GRP, color = Depth_Group)) +
    geom_line() +
    facet_wrap(~ Depth_Group, scales = "free_y") +
    scale_x_date(date_breaks = "15 days", date_labels = "%m/%d/%Y") +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black"),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.margin = margin(t = 10, r = 10, b = 40, l = 10)
    ) +
    labs(title = paste("Mean GRP Time Series for Node", node_id))
  
  # Print the plot
  print(p)
}

# Example usage:
# Assuming df_summary is your data frame
plot_node_grps(df_summary, node_id = 1)

# Need to do the same ggplot and .gif for Temp - adjust names + cbar breaks 
# ... but use viridis or diverging instead of brewer blues






# Need to do the same ggplot and .gif for Temp - adjust names + cbar breaks - change to Marks function 
# ... but use viridis or diverging instead of brewer blues










################################### Potentially useful non-cluster that ran quick ####################
for (file_idx in seq_along(files)) {
  if (!file.exists(files[file_idx])) {
    stop(paste("File does not exist:", files[file_idx]))
  }
  
  grp_data <- tryCatch(readRDS(files[file_idx]), error = function(e) NULL)
  
  if (is.null(grp_data)) {
    stop(paste("Failed to read file:", files[file_idx]))
  }
  
  # Check if grp_data contains finite values
  if (!all(is.finite(grp_data))) {
    stop(paste("Non-finite values in file:", files[file_idx]))
  }
  
  # Continue with processing...
}
# Assuming the validation passed and you want to continue with the processing:
for (file_idx in seq_along(files)) {
  grp_data <- readRDS(files[file_idx])
  
  # Assuming you have a predefined start_date
  num_days <- if (file_idx < length(files)) 30 else 7
  date_seq <- seq(start_date + sum(num_days[1:(file_idx - 1)]), by = "day", length.out = num_days)
  
  df_long <- melt(grp_data)
  colnames(df_long) <- c("Node", "Depth", "Time", "GRP")
  df_long$Date <- date_seq[df_long$Time]
  
  # Continue with your data processing
  # e.g., appending df_long to a list, combining data frames, etc.
  
  # Optional: Free up memory if needed
  #rm(grp_data, df_long)
  gc()
}
##############################################################



# Set working directory and define directories
main_dir <- "/Volumes/WD Backup/Rowe_FVCOM_data"
data_dir <- file.path(main_dir, "Model_output_2010")
frame_dir <- "animation_frames_2010"

#data<- readRDS("/Volumes/WD Backup/Rowe_FVCOM_data/Model_output_2010/erie_0010_GRP.rds")
# Ensure the directory exists
dir.create(frame_dir, showWarnings = FALSE)

# Define the start date
start_date <- as.Date("2010-03-31")

# List all the GRP files for the whole lake
files <- list.files(data_dir, pattern = paste0("erie_\\d{4}_GRP\\.rds$"), full.names = TRUE)
file_length<- length(files)

# Initialize an empty list to store data frames from each file
df_combined_all <- list()





# Parallel processing setup
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
clusterExport(cl, c("start_date", "files", "file_length", "melt", "as.Date", "seq"))

# Loop over files in parallel
# Loop over files in parallel with error handling
df_combined_all <- parLapply(cl, seq_along(files), function(file_idx) {
  tryCatch({
    grp_data <- readRDS(files[file_idx])
    
    # Calculate the start date for this file
    start_file_date <- start_date + sum(ifelse(1:(file_idx - 1) < file_length, 30, 7))
    
    # Determine the number of days in this file
    num_days <- if (file_idx < 10) 30 else 7
    
    # Generate the sequence of dates
    date_seq <- seq(start_file_date, by = "day", length.out = num_days)
    
    # Convert grp_data to long format
    df_long <- melt(grp_data)
    colnames(df_long) <- c("Node", "Depth", "Time", "GRP")
    
    # Assign the correct Date values based on the sequence
    df_long$Date <- date_seq[df_long$Time]
    
    return(df_long)
  }, error = function(e) {
    # Print an error message if something goes wrong
    cat("Error in processing file index", file_idx, ":", conditionMessage(e), "\n")
    return(NULL)
  })
})
# Stop the cluster
stopCluster(cl)
# Combine the results, removing any NULL entries
df_combined_all <- do.call(rbind, df_combined_all[!sapply(df_combined_all, is.null)])


# Memory management
rm(grp_data)
gc()

# Create Depth_Group variable
df_combined_all$Depth_Group <- cut(
  df_combined_all$Depth,
  breaks = c(0, 5, 10, 15, 20),
  labels = c("1-5", "6-10", "11-15", "16-20"),
  include.lowest = TRUE
)



# Calculate mean GRP by Node, Date, and Depth_Group
df_summary <- df_combined_all %>%
  group_by(Node, Date, Depth_Group) %>%
  summarize(Mean_GRP = mean(GRP, na.rm = TRUE), .groups = "drop")


# Memory management
rm(df_combined_all)
gc()


# Load spatial data files and functions
source("proj_functions.R")
source("GetColors_GRP.R")
load("erie05lcc_grid_polys.Rdata")
load("mask2.Rdata")
load("erie05lcc_grid_coords.Rdata")

### Creating subset of test frames to quickly assess this block
unique_dates = unique(df_summary$Date)
test_dates <- unique_dates[1:5]

# Calculate global colorbar limits from the entire dataset
values_all <- df_summary$Mean_GRP
min_value <- min(df_summary$Mean_GRP, na.rm = TRUE)
min_val<- min_value - 0.01
max_value <- max(df_summary$Mean_GRP, na.rm = TRUE)
max_val <- max_val +0.01
interval <- 0.005  # Define your desired interval

# Use the actual min and max values to create the breaks
global_brks <- seq(min_val, max_val, by = interval)
global_cols <- getColorsVal(values = values_all, brk = global_brks, num_breaks = nlevels(global_brks))  # Get colors based on global breaks

# Correct layout with 6 panels: 4 for depth groups, 2 for colorbars
layout_matrix <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
#show(layout_matrix)


# Set up a GIF output
saveGIF({
  
  # Loop over each unique date to generate frames
  for (date in unique_dates) {                     #unique_dates
    test_data <- df_summary %>% filter(Date == date)
    
    # Initialize a new plot
    plot.new()
    
    # Set up a layout for the plots with two shared colorbars
    layout(layout_matrix, widths = c(1, 1, 0.25), heights = c(1, 1))
    
    # Define the panels where the depth groups will be placed
    panel_indices <- c(1, 2, 4, 5)
    
    # Plot for each of the first four panels (depth groups)
    for (i in seq_along(panel_indices)) {
      # Set the panel to plot in
      par(mfg = c(floor((panel_indices[i] - 1) / 3) + 1, (panel_indices[i] - 1) %% 3 + 1))
      
      depth_group <- unique(test_data$Depth_Group)[i]
      depth_data <- test_data %>% filter(Depth_Group == depth_group)
      
      # Calculate the GRP values for coloring
      values <- depth_data$Mean_GRP
      brks <- global_brks  # Use global breaks
      
      # Assuming getColorsVal is a valid function that returns a list with 'cols' and 'zlim'
      cols <- getColorsVal(values = values, brk = brks)  # Get colors based on breaks
      
      # Plotting using base R functions
      par(mar = c(4, 0, 5, 0))  # Negative left margin  # Adjust margins to shift plot to the left
      plotPolys(shoreline, col = "white", projection = TRUE,
                xlim = xlim, ylim = ylim,
                xlab = "", ylab = "", xaxt = "n", yaxt = "n")
      
      # Add the points with the appropriate color and size
      points(coordsn$X / 1000, coordsn$Y / 1000, pch = 15, col = cols$cols, cex = 1.405 ) #1.4 original for scale (1K)
      points(coordsn$X / 1000, coordsn$Y / 1000, pch = 15, col = cols$cols, cex = 1.05 )  #1.0 original for scale (800)
      
      # Plot the land mask to cover areas outside the model domain
      addPolys(mask2, col = "tan")
      
      # Add a title to the plot
      title(main = paste("Mean GRP on", format(as.Date(date), "%Y-%m-%d"), "- Depth Group (sigma layer):", depth_group), cex.main = 2)
    }
    
    
    # Plot the colorbars in panels 3 and 6
    for (panel in c(3, 6)) {
      # Set the panel to plot in
      par(mfg = c(floor((panel - 1) / 3) + 1, (panel - 1) %% 3 + 1))
      
      par(mar = c(5, 0, 5, 9.5))  # Adjust margins for colorbars
      lab.breaks <- sprintf("%1.3f", global_brks)  # Format the breaks for labels
      
      # Set up the z matrix for the colorbar
      z <- matrix(seq(min(global_cols$zlim), max(global_cols$zlim), length.out = length(global_cols$colorbar)), nrow = 1)
      
      # Plot the color scale
      image(seq(0.01, 0.02, length.out = 2), seq_along(global_cols$colorbar), z, col = global_cols$colorbar, axes = FALSE, xlab = "", ylab = "")
      axis(4, at = seq_along(lab.breaks), labels = lab.breaks, las = 2, cex.axis = 1.9, font = 2, tick = FALSE)
      
      # Add a label to the colorbar
      label <- expression(paste("GRP ", g, g^-1, d^-1))
      mtext(label, side = 3, line = 1, cex = 1.2)
    }
    
  }
  
}, movie.name = "GRP_Animation_2010_viridis.gif", interval = 0.25, ani.width = 1250, ani.height = 900, autobrowse = FALSE)


