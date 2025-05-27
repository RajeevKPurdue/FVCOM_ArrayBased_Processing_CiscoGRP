#!/usr/bin/env python3
"""
Simple FVCOM NetCDF Plotter

This script provides a simple GUI to visualize temperature and dissolved oxygen
data from FVCOM NetCDF files. It allows you to select files using dialog boxes
and create plots without having to type file paths.

Usage:
    Simply run this script and follow the dialog prompts.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import pandas as pd
from tkinter import Tk, filedialog, messagebox, simpledialog
from matplotlib.widgets import Slider
import matplotlib

matplotlib.use('TkAgg')  # Use Tk backend for interactive displays

def main():
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()

    # Ask user to select the NetCDF file
    print("Please select a NetCDF file...")
    file_path = filedialog.askopenfilename(
        title="Select NetCDF file",
        filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")]
    )

    if not file_path:
        messagebox.showerror("Error", "No file selected. Exiting.")
        root.destroy()
        exit()

    print(f"Selected file: {file_path}")

    # Try to find grid file in the same directory
    grid_file = os.path.join(os.path.dirname(file_path), 'grid.nc')
    print(f"Looking for grid file at: {grid_file}")

    # Open the NetCDF file
    try:
        ds = xr.open_dataset(file_path)
        print("Successfully opened NetCDF file")
        print(f"Available variables: {list(ds.data_vars)}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file: {str(e)}")
        root.destroy()
        exit()

    # Check if necessary variables exist
    required_vars = ['temperature', 'dissolved_oxygen', 'depth']
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        print(f"Warning: Some expected variables are missing: {missing_vars}")

    # Get available dates
    dates = pd.to_datetime(ds.time.values)
    date_strings = [date.strftime('%Y-%m-%d') for date in dates]
    print(f"Found {len(dates)} dates in the file")

    # Let user choose which date to plot
    date_list = "\n".join([f"{i}: {date}" for i, date in enumerate(date_strings[:10])])
    if len(dates) > 10:
        date_list += "\n..."

    date_index = simpledialog.askinteger(
        "Select Date",
        f"Enter date index (0-{len(dates) - 1}):\n{date_list}",
        minvalue=0, maxvalue=len(dates) - 1
    )

    if date_index is None:
        date_index = 0
        print(f"Using default date index: 0 ({date_strings[0]})")
    else:
        print(f"Selected date index: {date_index} ({date_strings[date_index]})")

    # Let user choose which variable to plot
    var_options = ["temperature", "dissolved_oxygen"]
    available_vars = [var for var in var_options if var in ds.data_vars]

    if not available_vars:
        messagebox.showerror("Error", "Neither temperature nor dissolved oxygen variables found in file")
        ds.close()
        root.destroy()
        exit()

    variable = simpledialog.askstring(
        "Select Variable",
        f"Enter variable to plot ({', '.join(available_vars)}):",
        initialvalue=available_vars[0]
    )

    if variable not in available_vars:
        variable = available_vars[0]
        print(f"Using default variable: {variable}")
    else:
        print(f"Selected variable: {variable}")

    # Try to get coordinates
    try:
        # First try grid file
        if os.path.exists(grid_file):
            print("Loading coordinates from grid file...")
            grid_ds = xr.open_dataset(grid_file)
            element_lon = grid_ds.element_lon.values
            element_lat = grid_ds.element_lat.values
            grid_ds.close()
            print("Using coordinates from grid file")
        # Then try data file
        elif 'element_lon' in ds.variables and 'element_lat' in ds.variables:
            print("Loading coordinates from data file...")
            element_lon = ds.element_lon.values
            element_lat = ds.element_lat.values
            print("Using coordinates from data file")
        else:
            messagebox.showerror("Error", "Could not find element coordinates in grid or data file")
            ds.close()
            root.destroy()
            exit()
    except Exception as e:
        messagebox.showerror("Error", f"Error loading coordinates: {str(e)}")
        ds.close()
        root.destroy()
        exit()

    # Get variable data
    var_data = ds[variable].values[date_index]
    var_units = ds[variable].attrs.get('units', '')
    var_name = 'Temperature' if variable == 'temperature' else 'Dissolved Oxygen'
    date_str = dates[date_index].strftime('%Y-%m-%d')

    # Get surface and bottom layers
    surface_data = var_data[:, 0]  # First sigma layer (surface)
    bottom_data = var_data[:, -1]  # Last sigma layer (bottom)

    # Create plot
    try:
        # Try to import cmocean for better colormaps
        try:
            import cmocean.cm as cmo
            cmap = cmo.thermal if variable == 'temperature' else cmo.oxy
        except ImportError:
            cmap = 'viridis' if variable == 'temperature' else 'cividis'
            print("Note: Install cmocean package for better colormaps")

        # Create the figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Get common color scale
        vmin = min(np.nanmin(surface_data), np.nanmin(bottom_data))
        vmax = max(np.nanmax(surface_data), np.nanmax(bottom_data))

        # Plot surface layer
        sc0 = axes[0].scatter(element_lon, element_lat, c=surface_data,
                              cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.8)
        axes[0].set_title(f'Surface {var_name} - {date_str}')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_aspect('equal')

        # Plot bottom layer
        sc1 = axes[1].scatter(element_lon, element_lat, c=bottom_data,
                              cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.8)
        axes[1].set_title(f'Bottom {var_name} - {date_str}')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_aspect('equal')

        # Add colorbar
        cbar = fig.colorbar(sc0, ax=axes, orientation='horizontal', pad=0.05)
        cbar.set_label(f'{var_name} ({var_units})')

        plt.tight_layout()

        # Ask if user wants to save the figure
        save_choice = messagebox.askyesno("Save Figure", "Would you like to save this figure?")
        if save_choice:
            output_dir = filedialog.askdirectory(title="Select directory to save figure")
            if output_dir:
                filename = f"{os.path.basename(file_path).split('.')[0]}_{date_str}_{variable}_surface_bottom.png"
                fig_path = os.path.join(output_dir, filename)
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Figure saved to {fig_path}")

        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Error creating plot: {str(e)}")

    finally:
        # Close dataset
        ds.close()
        root.destroy()


def create_interactive_plot():
    """
    Create an interactive plot with a time slider
    """
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()

    # Ask user to select the NetCDF file
    file_path = filedialog.askopenfilename(
        title="Select NetCDF file",
        filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")]
    )

    if not file_path:
        messagebox.showerror("Error", "No file selected. Exiting.")
        root.destroy()
        exit()

    # Try to find grid file in the same directory
    grid_file = os.path.join(os.path.dirname(file_path), 'grid.nc')

    # Open the NetCDF file
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file: {str(e)}")
        root.destroy()
        exit()

    # Let user choose which variable to plot
    var_options = ["temperature", "dissolved_oxygen"]
    available_vars = [var for var in var_options if var in ds.data_vars]

    if not available_vars:
        messagebox.showerror("Error", "Neither temperature nor dissolved oxygen variables found in file")
        ds.close()
        root.destroy()
        exit()

    variable = simpledialog.askstring(
        "Select Variable",
        f"Enter variable to plot ({', '.join(available_vars)}):",
        initialvalue=available_vars[0]
    )

    if variable not in available_vars:
        variable = available_vars[0]

    # Try to get coordinates
    try:
        if os.path.exists(grid_file):
            grid_ds = xr.open_dataset(grid_file)
            element_lon = grid_ds.element_lon.values
            element_lat = grid_ds.element_lat.values
            grid_ds.close()
        elif 'element_lon' in ds.variables and 'element_lat' in ds.variables:
            element_lon = ds.element_lon.values
            element_lat = ds.element_lat.values
        else:
            messagebox.showerror("Error", "Could not find element coordinates in grid or data file")
            ds.close()
            root.destroy()
            exit()
    except Exception as e:
        messagebox.showerror("Error", f"Error loading coordinates: {str(e)}")
        ds.close()
        root.destroy()
        exit()

    # Get variable data and metadata
    var_data = ds[variable].values
    var_units = ds[variable].attrs.get('units', '')
    var_name = 'Temperature' if variable == 'temperature' else 'Dissolved Oxygen'
    dates = pd.to_datetime(ds.time.values)

    # Try to import cmocean for better colormaps
    try:
        import cmocean.cm as cmo
        cmap = cmo.thermal if variable == 'temperature' else cmo.oxy
    except ImportError:
        cmap = 'viridis' if variable == 'temperature' else 'cividis'

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.25)  # Make room for the slider

    # Get initial data
    date_index = 0
    surface_data = var_data[date_index, :, 0]
    bottom_data = var_data[date_index, :, -1]

    # Get common color scale across all times
    vmin = np.nanmin(var_data[:, :, [0, -1]])
    vmax = np.nanmax(var_data[:, :, [0, -1]])

    # Create initial plots
    sc0 = axes[0].scatter(element_lon, element_lat, c=surface_data,
                          cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.8)
    axes[0].set_title(f'Surface {var_name}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_aspect('equal')

    sc1 = axes[1].scatter(element_lon, element_lat, c=bottom_data,
                          cmap=cmap, vmin=vmin, vmax=vmax, s=1.5, alpha=0.8)
    axes[1].set_title(f'Bottom {var_name}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_aspect('equal')

    # Add colorbar
    cbar = fig.colorbar(sc0, ax=axes, orientation='horizontal', pad=0.05)
    cbar.set_label(f'{var_name} ({var_units})')

    # Add date label
    date_label = fig.suptitle(f'Date: {dates[date_index].strftime("%Y-%m-%d")}', fontsize=14)

    # Add time slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Date Index',
        valmin=0,
        valmax=len(dates) - 1,
        valinit=0,
        valstep=1
    )

    # Update function for slider
    def update(val):
        date_index = int(slider.val)
        surface_data = var_data[date_index, :, 0]
        bottom_data = var_data[date_index, :, -1]

        sc0.set_array(surface_data)
        sc1.set_array(bottom_data)

        date_label.set_text(f'Date: {dates[date_index].strftime("%Y-%m-%d")}')

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.ion()  # Turn on interactive mode
    plt.show()

    # Close dataset
    ds.close()
    root.destroy()


if __name__ == "__main__":
    print("Simple FVCOM NetCDF Plotter")
    print("---------------------------")
    print("1. Basic Plot (surface and bottom layers)")
    print("2. Interactive Plot (with time slider)")
    print("---------------------------")

    choice = input("Choose an option (1/2): ")

    if choice == "2":
        create_interactive_plot()
    else:
        main()