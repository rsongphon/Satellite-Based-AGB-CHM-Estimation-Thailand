import os
import xarray as xr
import rasterio as rio
import argparse
import rioxarray
from threading import Lock
import sys
import scipy
import numpy as np

def get_imgs_and_profile(imgs_path):
    
    images = []
    profile = None
    print(f'Looking for {imgs_path}')
    for root, dirs, files in os.walk(imgs_path, topdown=True):
        #print(root)
        for fname in files:
            if fname.endswith('S2ABands_stk.tif'):
                print(fname)
                if profile is None:
                    temp_img = rio.open(os.path.join(root, fname))
                    profile = temp_img.profile
                temp = rioxarray.open_rasterio(os.path.join(root, fname), chunks="auto",lock=False)
                #temp.persist(scheduler="threads", num_workers=4)
                images.append(temp)
    
    return (images,profile)

def get_SCL_imgs_and_profile(imgs_path):
    
    images = []
    profile = None
    print(f'Looking for {imgs_path}')
    for root, dirs, files in os.walk(imgs_path, topdown=True):
        #print(root)
        for fname in files:
            if fname.endswith('SCL.jp2'):
                print(fname)
                if profile is None:
                    temp_img = rio.open(os.path.join(root, fname))
                    profile = temp_img.profile
                temp = rio.open(os.path.join(root, fname))
                img = temp.read()
                #temp.persist(scheduler="threads", num_workers=4)
                images.append(img)
    
    return (np.array(images).squeeze(),profile)

##########################################
#Median
##########################################
def get_median_img(imgs, no_of_bands):
    
    bands_medians = []
    for b in range(no_of_bands):
        bands = [img.sel(band=b+1) for img in imgs]
        bands_comp = xr.concat(bands, dim='band')
        bands_medians.append(bands_comp.median(dim='band', skipna=True))
        median = xr.concat(bands_medians,dim='band')
        #median = median.persist(scheduler="threads", num_workers=4)
    return median

##########################################
#Mode
##########################################

def _mode(a):
    vals = scipy.stats.mode(a, keepdims=True)
    # only return the mode (discard the count)
    return vals[0].squeeze()




def get_args():
    parser = argparse.ArgumentParser(description='Create Composite image from time-series raster')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute Image path of stacked time-series raster')
    parser.add_argument('--output-path', metavar='output_path', type=str, help='Output path of composite image(include filename)')
    parser.add_argument('--no-of-bands', metavar='no_of_bands', type=int, help='Number of band compostie')
    parser.add_argument('--mode', metavar='mode', type=str, help='Composite mode : median or mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Load image..')
    if args.mode == 'median':
        imgs, imgs_profile = get_imgs_and_profile(args.input_path)
        print('Get Median image..')
        median_img = get_median_img(imgs, args.no_of_bands)
        imgs_profile['crs'] = 32647
        median_img.attrs = imgs_profile
        print('Write Median image')

        median_img = median_img.rio.to_raster(args.output_path, tiled=True, lock=Lock(), compute=False)
        median_img.compute(scheduler="threads", num_workers=4)
        print("Done!")
    elif args.mode == 'mode':
        imgs, imgs_profile = get_SCL_imgs_and_profile(args.input_path)
        print(imgs.shape)
        print('Get Mode image..')
        mode_img = _mode(imgs)
        print('Write Mode image')
        print(mode_img.shape)
        with rio.open(args.output_path, "w", **imgs_profile) as dest:
            dest.write(mode_img)
        print("Done!")
    else:
        sys.exit("Wrong composite mode. Mode can be 'median' or 'mode' only") 
        
