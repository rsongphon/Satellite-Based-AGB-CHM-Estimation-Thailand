import argparse
import geowombat as gw


def get_args():
    parser = argparse.ArgumentParser(description='Clip value of Sentinel 2 and Normalize in range 0-1')
    parser.add_argument('--input-path',  metavar='input_path', type=str, help='Absolute Image path of composite stacked raster')
    parser.add_argument('--output-path', metavar='output_path', type=str, help='Output path of normalize image(include filename)')
    parser.add_argument('--value', metavar='value', type=int, help='Any value higher than this value will be set to this value')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Load image..')
    
    with gw.config.update(nodata=0 , bigtiff= 'YES'):
        with gw.open(args.input_path) as src:
            #  replace 0 with nan
            #src=src.gw.mask_nodata() 
            
            # Xarray drops attributes
            attrs = src.attrs.copy()

            # Apply operations on the DataArray
    
            # clip where data > 5000 to 5000
            src = src.where(src < args.value, args.value)
            
            #Normalize to range 0-1
            src = src / args.value

            src.attrs = attrs
            
            print(src.attrs)
            # Write the data to a GeoTiff
            src.gw.to_raster(args.output_path,
                        n_workers=4,    # number of process workers sent to ``concurrent.futures``
                        )   

