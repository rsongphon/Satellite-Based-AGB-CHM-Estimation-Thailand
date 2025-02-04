#!/bin/bash
tile=$1 
source /opt/conda/bin/activate dask_rasterio 
cd /home/jupyter/unet_canopyheight_estimation/src/features

# Composite median 
output_composite=$tile"_composite.tif"
python build_composite_image.py --input-path /home/jupyter/Sentinel2_Data/$tile --output-path /home/jupyter/Sentinel2_Data/Composite_median/$output_composite --no-of-bands 10 --mode median

# Cap value and Normalize image
input_comp_img=$tile"_composite.tif"
output_norm_img=$tile"_composite_norm.tif"
python preprocessing_sentinel2.py --input-path /home/jupyter/Sentinel2_Data/Composite_median/$input_comp_img --output-path /home/jupyter/Sentinel2_Data/Composite_Normalize/$output_norm_img --value 5000

# Annual NDVI
python build_ndvi_image.py --input-path /home/jupyter/Sentinel2_Data/$tile

output_max_NDVI=$tile"_max_NDVI.tif"
python build_max_annual_ndvi_image.py --input-path /home/jupyter/Sentinel2_Data/$tile --output-path /home/jupyter/Sentinel2_Data/Max_NDVI/$output_max_NDVI
#echo $output_max_NDVI
