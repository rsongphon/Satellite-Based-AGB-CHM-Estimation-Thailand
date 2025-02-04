import os
import xarray as xr
import rasterio as rio
import argparse
import rioxarray
from threading import Lock



def get_imgs_and_profile(imgs_path):
    
    images = []
    profile = None
    print(f'Looking for {imgs_path}')
    for root, dirs, files in os.walk(imgs_path, topdown=True):
        #print(root)
        for fname in files:
            if fname.endswith('S2ABands_NDVI.tif'):
                print(fname)
                if profile is None:
                    temp_img = rio.open(os.path.join(root, fname))
                    profile = temp_img.profile
                temp = rioxarray.open_rasterio(os.path.join(root, fname), chunks="auto",lock=False)
                #temp.persist(scheduler="threads", num_workers=4)
                images.append(temp)
    
    return (images,profile)


##########################################
# #########################################


def get_max_img(imgs):
    
    bands_max = []
    for b in range(1): # ndvi has 1 band
        bands = [img.sel(band=b+1) for img in imgs]
        bands_comp = xr.concat(bands, dim='band')
        bands_max.append(bands_comp.max(dim='band', skipna=True))
        max_ndvi = xr.concat(bands_max,dim='band')
        #median = median.persist(scheduler="threads", num_workers=4)
    return max_ndvi

def get_args():
    parser = argparse.ArgumentParser(description='Create Annual maximum NDVI image from time-series NDVI image')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute Image path of time-series NDVI image')
    parser.add_argument('--output-path', metavar='output_path', type=str, help='Output path of maximum NDVI image(include filename)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Load image..')
    imgs, imgs_profile = get_imgs_and_profile(args.input_path)
    print('Get Max NDVI image..')
    max_ndvi_img = get_max_img(imgs)
    imgs_profile['crs'] = 32647
    max_ndvi_img.attrs = imgs_profile
    print('Write Max NDVI image')
    
    max_ndvi_img = max_ndvi_img.rio.to_raster(args.output_path, tiled=True, lock=Lock(), compute=False)
    max_ndvi_img.compute(scheduler="threads", num_workers=4)
    print("Done!")
