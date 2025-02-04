import rasterio as rio
from shapely.geometry import box
import geopandas as gpd
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Create raster boundary extend shapefile GeoPackage from any raster folder with profile')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute Image path of all  raster file must end with ".tif"')
    parser.add_argument('--output-path',  metavar='output_path', type=str, help='Output directory of shapefile')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print('Load image..')
    
    for root, dirs, files in os.walk(args.input_path, topdown=True):
        #print(root)
        for fname in files:
            if fname.endswith('.tif'):
                #print(fname)
                ra = rio.open(os.path.join(root, fname))
                bounds  = ra.bounds
                geom = box(*bounds)
                df = gpd.GeoDataFrame({"id":1,"geometry":[geom]})
                df = df.set_crs('EPSG:32647')
                #print(df)
                output_path = os.path.join(args.output_path, fname.replace('.tif','_boundary_shp_32647.gpkg'))
                print(f'Create {output_path}')
                df.to_file(output_path,  driver="GPKG")
                #df = df.to_crs(4326)
                #output_path = os.path.join(args.output_path, fname.replace('.tif','_boundary_shp_latlon4326.gpkg'))
                #print(f'Create {output_path}')
                #df.to_file(output_path,  driver="GPKG")