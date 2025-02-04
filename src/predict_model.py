import argparse
import logging
import os
import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from models import UNet
from tqdm import tqdm
from rasterio.windows import Window
from collections import OrderedDict
from torchsummary import summary

###########sliding window approach with average between boundary ###########

def predict_img(net,
                window_img_numpy,
                device):
    net.eval()
    #window_img_numpy = np.nan_to_num(window_img_numpy, copy=True, nan=0)
    #print(window_img_numpy)
    #print(window_img_numpy.shape)
    img = torch.from_numpy(window_img_numpy)
    
    # Create a mask tensor to identify NaN values
    mask = torch.isnan(img)

    # Apply the mask to zero out the NaN values
    image_no_nan = img.clone()
    image_no_nan[mask] = 0.0
    
    image_no_nan = image_no_nan.unsqueeze(0) # batch
    image_no_nan = image_no_nan.to(device=device, dtype=torch.float32) ### Cuda memory leak

    with torch.no_grad():
        #predictions, log_variances  = net(image_no_nan).cpu()
        predictions, log_variances  = net(image_no_nan)
        variances = torch.exp(log_variances)
        #output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    predict_np = predictions[0].float().cpu().numpy()
    variances_np = variances[0].float().cpu().numpy()
    del log_variances
    del image_no_nan
    del predictions
    del variances
    #print(predict_np.mean())
    return (predict_np,variances_np)


def get_args():
    parser = argparse.ArgumentParser(description='Predict Canopy height or AGB from input sentinel 1 and sentinel 2 images')
    parser.add_argument('--model_path', '-m', metavar='PATH',
                        help='Specify the path in which the ensemble model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--window_size', '-w', metavar='WINDOW', default=2196, help='Define window size of the prediction.')
    parser.add_argument('--stride', '-s', metavar='STRIDE', default=2000, help='Stride of the sliding window')
    parser.add_argument('--channels', '-c', dest='num_ch', type=int, default=14, help='Number of input_channels')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')

    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT_PREDICT.tif'

    return args.output or list(map(_generate_name, args.input))



if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Define the sliding window size and stride
    WINDOW_SIZE = args.window_size
    STRIDE = args.stride

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=args.num_ch, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model_path}')
    

    logging.info(f'Load Model Best Weight!')
    model_path = os.path.join(args.model_path,f'best_weights.pt')
    
    
    logging.info(f'Using device {device}')

    net.to(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    #for key, value in state_dict.items():
    #    print(key)
    net.load_state_dict(checkpoint['model'], strict=False)
    #summary(net, (16, 224, 224))

    logging.info('Model loaded!')

    # create tempporary folder for store window reading
    temp_dirpath = './temp/'
    os.makedirs(temp_dirpath , exist_ok=True)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        
        # Open raster input 
        img = rasterio.open(filename)
        #print(img.shape)
        out_profile = img.profile.copy()
        out_profile.update(count=1)
        
        # window operation read/write
        out_filename = out_files[i]
        dst_pred = rasterio.open(out_filename+'_pred', 'w', **out_profile)
        dst_var = rasterio.open(out_filename + '_var', 'w', **out_profile)
        

        # Create empty numpy array to store the predictions and keep track of overlapping counts
        predictions = np.memmap(os.path.join(temp_dirpath,'predictions.mymemmap'), dtype='float32', mode='w+', shape=img.shape)
        #predictions = np.zeros(img.shape)
        predictions = np.expand_dims(predictions, axis=0)
        
        #print(predictions.shape)
        
        variances = np.memmap(os.path.join(temp_dirpath,'variances.mymemmap'), dtype='float32', mode='w+', shape=img.shape)
        # variances = np.zeros(img.shape)
        variances = np.expand_dims(variances, axis=0)
        
        #print(variances.shape)
        counts = np.memmap(os.path.join(temp_dirpath,'counts.mymemmap'), dtype='float32', mode='w+', shape=img.shape)
        counts = np.expand_dims(counts, axis=0)
 
        

        # Iterate through the image using the sliding window approach
        for i in tqdm(range(0, img.shape[0] - WINDOW_SIZE + 1, STRIDE)):
            for j in range(0, img.shape[1] - WINDOW_SIZE + 1, STRIDE):
                
                
                # Define the row and column indices for the window
                start_row = i
                end_row = i+WINDOW_SIZE
                start_col = j
                end_col = j+WINDOW_SIZE
                
                # Extract the current window
                window_range = rasterio.windows.Window(start_col, start_row, end_col - start_col, end_row - start_row)
                window = img.read(window=window_range)
                
                # Forward pass through the model to get predictions
                pred , var = predict_img(net=net,
                               window_img_numpy=window,
                               device=device)
                
                # Accumulate predictions and counts for overlapping regions
                predictions[:,i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] += pred 
                variances[:,i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] += var
                counts[:,i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] += 1
                del pred
                del var
                del window
                
        # Average the predictions by dividing with the corresponding counts
        predictions /= counts
        variances /= counts
        
        print("Write raster")
        dst_pred.write(predictions)
        dst_var.write(variances)
        
        logging.info(f'Mask saved to {out_filename}')
        
        del predictions
        del variances
        del counts
        
        # remove temporary file
        os.remove(os.path.join(temp_dirpath,'predictions.mymemmap')) 
        os.remove(os.path.join(temp_dirpath,'variances.mymemmap')) 
        os.remove(os.path.join(temp_dirpath,'counts.mymemmap')) 

        dst_pred.close()
        dst_var.close()
        img.close()