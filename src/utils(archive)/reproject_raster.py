import os
import argparse
from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio

def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def get_args():
    parser = argparse.ArgumentParser(description='Reproject and align all of raster input with their respective geo-reference')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute directory path of raster input file . All file must start with (tilename_) and end with .tif')
    parser.add_argument('--ref-path',  metavar='ref_path', type=str, help='Absolute directory path of raster reference path to get geo-reference for each image. All file must start with (tilename_) and end with .tif')
    parser.add_argument('--output-path', metavar='output_path', type=str, help='Output Directory of alignment-reproject raster')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Load input image..')
    for root, dirs, files in os.walk(args.input_path, topdown=True):
        #print(root)
        # Looking in input file
        for fname in files:
            found = False
            if fname.endswith('.tif'):
                
                print(fname)
                # Extract tilename
                tilename =  fname.split('_')[0] # Ex: 47PMS
                print(tilename)
                
                input_path = os.path.join(root,fname)
                                    
                # Find match target raster 
                for root_target, dirs_target, files_target in os.walk(args.ref_path, topdown=True):
                    if found:
                        break
                    for fname_target in files_target:
                        if fname_target.endswith('.tif') and fname_target.startswith(tilename):
                            found = True
                            target_path = os.path.join(root_target,fname_target)
                            print(f'Match {target_path}')
                            break
                output_path = os.path.join(args.output_path,fname.replace('.tif','_repro_align.tif'))
                

                # co-register LS to match precip raster
                reproj_match(infile = input_path, 
                             match= target_path,
                             outfile = output_path)
    print('Done!')


                
