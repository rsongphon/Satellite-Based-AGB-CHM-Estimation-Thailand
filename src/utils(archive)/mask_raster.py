import os
import rasterio
import argparse
from rasterio.mask import mask
import geopandas as gpd




def get_args():
    parser = argparse.ArgumentParser(description='Mask raster from tile boundary shapefile')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute Image path of shapefile Geopackage. Filename must start with (tilename_)')
    parser.add_argument('--target-path',  metavar='target_path', type=str, help='Absolute  path of target raster to search and mask. All file must end with .tif')
    parser.add_argument('--output-path', metavar='output_path', type=str, help='Output Directory of mask raster')
    parser.add_argument('--crs', metavar='crs', type=int, default=4326, help='Coordinate use in both shapefile and raster default to EPSG:4326 (type int)')
    parser.add_argument('--target-name', metavar='target_name', type=str, help='Name for output file')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Load image..')
    for root, dirs, files in os.walk(args.input_path, topdown=True):
        #print(root)
        for fname in files:
            mask_done = False
            if fname.endswith('.gpkg'):
                print(fname)
                crop_extent = gpd.read_file(os.path.join(root, fname))
                
                #convert to desire crs
                crop_extent = crop_extent.to_crs(args.crs)
                print(crop_extent)
                
                # Get boundary extend
                shapes = [crop_extent.loc[row,'geometry'] for row in range(len(crop_extent))]
                
                print('Search ESA Macro tiles')
                # search and mask boundary
                for root_esa, dirs_esa, files_esa in os.walk(args.target_path, topdown=True):
                    if mask_done == True:
                        break
                    for fname_esa in files_esa:
                        if fname_esa.endswith('.tif'):
                            print(fname_esa)
                            try:
                                with rasterio.open(os.path.join(root_esa, fname_esa)) as src:
                                    out_image, out_transform = mask(src, shapes, crop=True)
                                    out_meta = src.meta
                                                   
                                    out_meta.update({"driver": "GTiff",
                                     "height": out_image.shape[1],
                                     "width": out_image.shape[2],
                                     "transform": out_transform})
                                print(f'Found in {fname_esa}')
                                output_filename = os.path.join(args.output_path,fname.split('_')[0]+f'_{args.target_name}_mask_.tif')
                                print(f'Writing {output_filename}')
                                with rasterio.open(output_filename, "w", **out_meta) as dest:
                                    dest.write(out_image)
                                mask_done = True
                                break
                            except Exception as e: 
                                print(e)
                                continue
                    
