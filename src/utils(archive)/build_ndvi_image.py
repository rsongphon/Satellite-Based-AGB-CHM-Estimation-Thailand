import os
import xarray as xr
import rasterio as rio
import argparse
import rioxarray
from threading import Lock
import geowombat as gw



def get_imgs_and_profile(imgs_path):
    
    images = []
    image_names = []
    profile = None
    print(f'Looking for {imgs_path}')
    for root, dirs, files in os.walk(imgs_path, topdown=True):
        #print(root)
        for fname in files:
            if fname.endswith('S2ABands_stk.tif'):
                #print('fname')
                if profile is None:
                    temp_img = rio.open(os.path.join(root, fname))
                    profile = temp_img.profile
                img_name = os.path.join(root, fname)
                temp = rioxarray.open_rasterio(img_name, chunks="auto",lock=False)
                #temp.persist(scheduler="threads", num_workers=4)
                images.append(temp)
                image_names.append(img_name)
    return (images,image_names,profile)


##########################################
# #########################################


def get_ndvi_img(imgs):
    ndvi_imgs = []
    #By default, the bands will be named by their index position (starting at 1)
    # Band 4 – Red = index 3
    # Band 8 – NIR = index 7
    for img in imgs:
        ndvi_imgs.append(img.gw.norm_diff(3, 7))
    
    return ndvi_imgs

def get_args():
    parser = argparse.ArgumentParser(description='Create individual NDVI image stacked raster')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute Image path of all stacked raster file must end with "S2ABands_stk.tif"')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Load image..')
    imgs,image_names, imgs_profile = get_imgs_and_profile(args.input_path)
    print('Get NDVI image..')
    NDVI_imgs = get_ndvi_img(imgs)
    imgs_profile['crs'] = 32647
    print('Write NDVI image')
    
    for ndvi_img,name in zip(NDVI_imgs,image_names):
        print(f'Write image '+ name.replace('stk','NDVI'))
        ndvi_img.attrs = imgs_profile
        ndvi = ndvi_img.rio.to_raster(name.replace('stk','NDVI'), tiled=True, lock=Lock(), compute=False)
        ndvi.compute(scheduler="threads", num_workers=4)
    print("Done!")
