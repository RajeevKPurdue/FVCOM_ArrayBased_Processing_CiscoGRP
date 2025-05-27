import geopandas as gpd

gdf = gpd.read_file('/Volumes/WD Backup/Erie_shp_files/lake_erie_fvcom_grid.shp')
print(gdf.crs)

#inhouse_gdf = gpd.read_file('/Users/rajeevkumar/Documents/Labs/Code_scripts/lake_erie_fvcom_grid.dbf')
#print(inhouse_gdf.crs)
# Choose a suitable projected CRS, for instance UTM zone 17N
#gdf_projected = gdf.to_crs("EPSG:32617")

# Compute the polygon areas in square meters
#gdf_projected["area_m2"] = gdf_projected.geometry.area

# The column area_m2 now contains accurate areas in square meters.
#gdf_projected.to_file("lake_erie_fvcom_grid_projected.shp")
