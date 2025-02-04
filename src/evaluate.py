import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import R2Score , MeanSquaredError
import torch.nn as nn
import numpy as np
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    loss_total = 0
    pred_data = torch.Tensor().to(device)
    true_data = torch.Tensor().to(device)
    mse_total = 0
    r2score = R2Score().to(device)
    mean_squared_error = MeanSquaredError().to(device)
    criterion = nn.L1Loss()
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            masks_pred = net(image)

            # calculate on only with GEDI pixel that is not nan
            masks_pred = masks_pred[~np.isnan(true_masks.cpu().numpy())]
            true_masks = true_masks[~np.isnan(true_masks.cpu().numpy())].float()
            
            pred_data = torch.cat((pred_data, masks_pred))
            true_data = torch.cat((true_data, true_masks))
            
            # MAE
            loss = criterion(masks_pred, true_masks)
            loss_total = loss_total + loss
            
            # mse
            mse = mean_squared_error(masks_pred, true_masks)
            mse_total = mse_total + mse
            
    net.train()
    
    mae_return = loss_total / max(num_val_batches, 1)
    r2_return = r2score(pred_data, true_data)
    mse_return = mse_total / max(num_val_batches, 1)
    return (mae_return,r2_return,mse_return)
