import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from utils.loss import RMSELoss, MELoss, GaussianNLL, LaplacianNLL
from pathlib import Path
from tqdm import tqdm
import os
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sen1Sen2_GEDI(torch.utils.data.Dataset):
    """Sentinel1_Sentinel2_GEDI_dataset."""

    def __init__(self, input_dir , target_dir):
        """
        Args:
        """
        self.input_dir = input_dir
        self.target_dir = target_dir 
        self.files = os.listdir(input_dir) # input and target has the same filename so os.listdir will produce same result
        
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.input_dir,
                                     self.files[idx])
        image = np.load(img_path)
        target_path = os.path.join(self.target_dir,
                                     self.files[idx])
        target_img = np.load(target_path)
        sample = {'image': image, 'mask': torch.from_numpy(target_img).float()}
        return    sample

class Trainer:

    def __init__(self, model, log_dir, args):
        """
        Initialize a Trainer object to train and test the model.
        Args:
            model: pytorch model
            log_dir: path to directory to save tensorboard logs
            args: argparse object (see setup_parser() in utils.parser)
        """
        
        
        self.model = model
        self.args = args
        
        # Skip training or not
        if self.args.skip_training:
            # Skip training -> don't care about train_mode
            # Do not initialize wandb
            self.out_dir = args.out_dir
            self.writer = SummaryWriter(log_dir=log_dir)
            # not using train dataset
            self.ds_train, self.ds_val, self.ds_test = self._setup_dataset()
            self.error_metrics = self._setup_metrics()
            
            
        else: 
            # Normal training
            if self.args.train_mode == 'resume':
                print('Continue training')
                self.args.train_mode = True
            else:
                print('Start training from sratch')
                self.args.train_mode = 'allow'
            
            # (Initialize logging)
            self.experiment = wandb.init(project=self.args.wabdb_name, resume=self.args.train_mode, anonymous='must')
            self.experiment.config.update(
            dict(epochs=self.args.nb_epoch, batch_size=self.args.batch_size, learning_rate=self.args.base_learning_rate, save_checkpoint=self.args.save_checkpoint, amp=self.args.amp))
            
            self.out_dir = args.out_dir
            self.writer = SummaryWriter(log_dir=log_dir)

            self.ds_train, self.ds_val, self.ds_test = self._setup_dataset()
            self.step_size = int(2*(len(self.ds_train) / self.args.batch_size))
            
            self.optimizer  = self._setup_optimizer()
            self.scheduler = self._setup_scheduler()
            
            ## Continue training / inference
            if wandb.run.resumed:
                # load best model weights
                print('ATTENTION: loading pretrained model weights from:')
                print(self.args.model_weights_path)
                self.checkpoint = torch.load(self.args.model_weights_path , map_location=device)
            
                self.model.load_state_dict(self.checkpoint['model'])
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                self.scheduler.load_state_dict(self.checkpoint['lr_sched'])

                # Get the starting epoch number from the resumed run
                self.start_epoch = self.checkpoint['epoch']
            else:
                self.start_epoch = 0
            
            self.grad_scaler = self._setup_grad_scaler()
            self.error_metrics = self._setup_metrics()
            self.global_step = 0

    def _setup_metrics(self):
        error_metrics = {'MSE': torch.nn.MSELoss(),
                         'RMSE': RMSELoss(),
                         'MAE': torch.nn.L1Loss(),
                         'ME': MELoss(),
                         'gaussian_nll':GaussianNLL(),
                         'laplacian_nll':LaplacianNLL()}


        print('error_metrics.keys():', error_metrics.keys())
        error_metrics['loss'] = error_metrics[self.args.loss_key]
        return error_metrics
    
    def _setup_grad_scaler(self):
        return torch.cuda.amp.GradScaler(enabled=self.args.amp)
    
    def _setup_optimizer(self):
        if self.args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.base_learning_rate,
                                        weight_decay=self.args.l2_lambda)
            
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.base_learning_rate,
                                        weight_decay=self.args.l2_lambda , momentum=self.args.momentum , foreach = True)
            
        else:
            raise ValueError("Solver '{}' is not defined.".format(self.args.optimizer))
        
        return optimizer
    
    def _setup_scheduler(self):
        if self.args.optimizer == 'ADAM':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=self.args.base_learning_rate ,max_lr=0.1,step_size_up=self.step_size, mode='triangular2',cycle_momentum=False)
            
        elif self.args.optimizer == 'SGD':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=self.args.base_learning_rate ,max_lr=0.1,step_size_up=self.step_size, mode='triangular2',cycle_momentum=False)
            
        else:
            raise ValueError("Solver '{}' is not defined.".format(self.args.optimizer))
        
        return scheduler


    def _setup_dataset(self):
        
        # 1. Create dataset
        ds_train = Sen1Sen2_GEDI(input_dir = Path(os.path.join(self.args.train_input_data_dir,'train')) , target_dir = Path( os.path.join(self.args.train_label_data_dir,'train')))
        ds_val = Sen1Sen2_GEDI(input_dir = Path(os.path.join(self.args.train_input_data_dir,'val'))  , target_dir = Path(os.path.join(self.args.train_label_data_dir,'val')))
        ds_test = Sen1Sen2_GEDI(input_dir = Path(os.path.join(self.args.train_input_data_dir,'test'))  , target_dir = Path(os.path.join(self.args.train_label_data_dir,'test')) )
        
        return ds_train, ds_val , ds_test
    
    def count_infinite_elements(self, x):
        return torch.sum(torch.logical_not(torch.isfinite(x))).item()
    
    def train(self):
        """
        A routine to train and validated the model for several epochs.
        """
        print(self.args.batch_size)
        # Initialize train and validation loader
        dl_train = DataLoader(self.ds_train, batch_size=self.args.batch_size, shuffle=True, drop_last=True ,num_workers=self.args.num_workers)
        dl_val = DataLoader(self.ds_val, batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=self.args.num_workers)

        # Init best losses for weights saving.
        loss_val_best = np.inf
        best_epoch = None
        

        print('Starting training')

        # Start training
        for epoch in range(self.start_epoch,self.args.nb_epoch):
            epoch += 1
            print('Epoch: {} / {} '.format(epoch, self.args.nb_epoch))

            # optimize parameters
            training_metrics = self.optimize_epoch(dl_train , epoch)
            # validated performance
            val_dict, val_metrics = self.validate(dl_val)

            # -------- LOG TRAINING METRICS --------
            metric_string = 'TRAIN: '
            for metric in self.error_metrics.keys():
                # tensorboard logs
                self.writer.add_scalar('{}/train'.format(metric), training_metrics[metric], epoch)
                self.writer.add_scalar('learning_rate',  self.optimizer.param_groups[0]['lr'] , epoch)
                metric_string += ' {}: {:.3f},'.format(metric, training_metrics[metric])
            print(metric_string)

            # -------- LOG VALIDATION METRICS --------
            metric_string = 'VAL:   '
            for metric in self.error_metrics:
                # tensorboard logs
                self.writer.add_scalar('{}/val'.format(metric), val_metrics[metric], epoch)
                metric_string += ' {}: {:.3f},'.format(metric, val_metrics[metric])
            print(metric_string)

            # logging the estimated variance
            if 'log_variances' in val_dict:
                val_dict['variances'] = torch.exp(val_dict['log_variances'])


                self.writer.add_scalar('var_mean/val', torch.mean(val_dict['variances']), epoch)
                self.writer.add_scalar('std_mean/val', torch.mean(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('std_min/val', torch.min(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('std_max/val', torch.max(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('var_count_infinite_elements/val', self.count_infinite_elements(val_dict['variances']), epoch)

                print('VAL: Number of infinite elements in variances: ', self.count_infinite_elements(val_dict['variances']))

            if val_metrics['loss'] < loss_val_best:
                loss_val_best = val_metrics['loss']
                best_epoch = epoch
                # save and overwrite the best model weights:
                path = Path(self.out_dir) / 'best_checkpoint.pt'

                checkpoint = { 
                                'epoch': best_epoch,
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'lr_sched': self.scheduler.state_dict()}#'lr_sched': self.scheduler.state_dict()
                
                torch.save(checkpoint, path)

                print('Saved weights at {}'.format(path))

            # stop training if loss is nan
            if np.isnan(training_metrics['loss']) or np.isnan(val_metrics['loss']):
                raise ValueError("Training loss is nan. Stop training.")

        # Save Checkpoint weight
            if self.args.save_checkpoint:
                path_checkpoint = Path(self.args.all_checkpoint_dir)
                
                path_checkpoint.mkdir(parents=True, exist_ok=True)
                
                checkpoint = { 
                                'epoch': epoch,
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'lr_sched': self.scheduler.state_dict()}#'lr_sched': self.scheduler.state_dict()
                
                torch.save(checkpoint, os.path.join(path_checkpoint , 'checkpoint_epoch{}.pth'.format(epoch)))
                
                print(f'Checkpoint {epoch} saved!')
                
        print('Best val loss: {} at epoch: {}'.format(loss_val_best, best_epoch))


    def optimize_epoch(self, dl_train , epoch_num):
        """
        Run the optimization for one epoch.
        Args:
            dl_train: torch dataloader with training data.
        Returns: Dict with error metrics on training data (including the loss). Used for tensorboard logs.
        """
        # init running error
        training_metrics = {}
        for metric in self.error_metrics:
            training_metrics[metric] = 0

        total_count_infinite_var = 0

        # set model to training mode
        self.model.train()
        with tqdm(total=len(self.ds_train), desc=f'Epoch {self.args.nb_epoch}/{self.args.nb_epoch}', unit='img') as pbar:
            for batch in dl_train:
                images, true_masks = batch['image'], batch['mask']
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32) 
                
                assert images.shape[1] == self.args.num_ch, \
                    f'Network has been defined with {self.args.num_ch} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # Run forward pass

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=self.args.amp):

                    predictions, log_variances = self.model(images)
                    #print(predictions.size())
                    #predictions, log_variances = predictions[:, 0], predictions[:, 1]
                

                    # calculate on only with GEDI pixel that is not nan
                    predictions = predictions[~np.isnan(true_masks.detach().cpu().numpy())]
                    log_variances = log_variances[~np.isnan(true_masks.detach().cpu().numpy())]
                    true_masks = true_masks[~np.isnan(true_masks.detach().cpu().numpy())].float()

                    # pass predicted mean and log_variance to e.g. gaussian_nll
                    loss = self.error_metrics['loss'](predictions, log_variances, true_masks)

                    # debug
                    variances = torch.exp(log_variances)
                    count_infinite = self.count_infinite_elements(variances)
                    total_count_infinite_var += count_infinite

                

                # Run backward pass
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.scheduler.step()

                
                # compute metrics on every batch and add to running sum
                for metric in self.error_metrics:
                    if metric in ['gaussian_nll', 'laplacian_nll', 'loss']:
                        training_metrics[metric] += self.error_metrics[metric](predictions, log_variances, true_masks).item()
                    else:
                        training_metrics[metric] += self.error_metrics[metric](predictions, true_masks).item()
                            
                    
                pbar.update(images.shape[0])
                self.global_step  += 1
                #### to do ######
                self.experiment.log({
                    'train loss': loss.item(),
                    'learning rate': self.optimizer.param_groups[0]['lr'],
                    'step': self.global_step,
                    'epoch': epoch_num
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})


        # debug
        if total_count_infinite_var > 0:
            print('TRAIN DEBUG: ATTENTION: count infinite elements in variances is: {}'.format(total_count_infinite_var))

        # average over number of batches
        for metric in self.error_metrics.keys():
            training_metrics[metric] /= len(dl_train)
        return training_metrics

    def validate(self, dl_val):
        """
        Validate the model on validation data.
        Args:
            dl_val: torch dataloader with validation data
        Returns:
            val_dict: Dict with torch tensors for 'predictions', 'targets', 'log_variances'.
            val_metrics: Dict with error metrics on validation data (including the loss). Used for tensorboard logs.
        """
        # set model to eval model
        self.model.eval()

        # init validation results for current epoch
        val_dict = {'predictions': np.array([]), 'targets': np.array([])}

        val_dict['log_variances'] = np.array([])

        # iterate over the validation set
        with torch.no_grad():
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=self.args.amp):
                for batch in tqdm(dl_val, total=len(dl_val), desc='Validation round', unit='batch', leave=False):
                    
                    image, mask_true = batch['image'], batch['mask']
                    
                    inputs = image.to(device)
                    labels = mask_true.to(device)
                    
                    # predict the mask
                    predictions, log_variances = self.model(inputs)

                    #predictions, log_variances = predictions[:, 0], predictions[:, 1]
                    
                    # calculate on only with GEDI pixel that is not nan
                    predictions = predictions[~np.isnan(mask_true.detach().cpu().numpy())]
                    log_variances = log_variances[~np.isnan(mask_true.detach().cpu().numpy())]
                    labels = labels[~np.isnan(mask_true.detach().cpu().numpy())].float()
                    
                    val_dict['log_variances'] = np.concatenate((val_dict['log_variances'],log_variances.detach().cpu().numpy()))
                    val_dict['predictions'] = np.concatenate((val_dict['predictions'],predictions.detach().cpu().numpy())) 
                    val_dict['targets'] = np.concatenate((val_dict['targets'],labels.detach().cpu().numpy()))  
                for key in val_dict.keys():
                    #if val_dict[key]:
                    val_dict[key] = torch.from_numpy(val_dict[key])#torch.stack(val_dict[key], dim=0)
                    print("val_dict['{}'].shape: ".format(key), val_dict[key].shape)

        val_metrics = {}


        for metric in self.error_metrics:
            if metric in ['gaussian_nll', 'laplacian_nll', 'loss']:
                loss = self.error_metrics[metric](val_dict['predictions'],
                                                                 val_dict['log_variances'],
                                                                 val_dict['targets']).item()
                ## wandb logging
                try:
                    self.experiment.log({
                        'learning rate': self.optimizer.param_groups[0]['lr'],
                         metric : loss,
                        'step': self.global_step,
                        #'epoch': epoch_num
                            })
                except:
                    pass
                val_metrics[metric] = loss
            else:
                loss = self.error_metrics[metric](val_dict['predictions'],
                                                                     val_dict['targets']).item()
                                ## wandb logginh
                try:
                    self.experiment.log({
                        'learning rate': self.optimizer.param_groups[0]['lr'],
                         metric : loss,
                        'step': self.global_step,
                        #'epoch': epoch_num
                            })
                except:
                    pass
                
                val_metrics[metric] = loss
                    
        return val_dict, val_metrics
    


    def test(self, model_weights_path=None, dl_test=None):
        """
        Test trained model on test data.
        Args:
            model_weights_path: path to trained model weights. Default: "best_weights.pt"
            dl_test: torch dataloader with test data. Default: self.ds_test is loaded.
        Returns:
            test_metrics: Dict with error metrics on test data (including the loss). Used for tensorboard logs.
            test_dict: Dict with torch tensors for 'predictions', 'targets', 'variances'.
            metric_string: formatted string to print test metrics.
        """
        if dl_test is None:
            dl_test = DataLoader(self.ds_test, batch_size=self.args.batch_size, shuffle=False,  drop_last=True , num_workers=self.args.num_workers)
        # test performance

        if model_weights_path is None:
            model_weights_path = Path(self.out_dir) / 'best_weights.pt'

        # load best model weights
        self.model.load_state_dict(torch.load(model_weights_path))

        test_dict, test_metrics = self.validate(dl_test)

        # convert log(var) to var
        test_dict['variances'] = torch.exp(test_dict['log_variances'])
        del test_dict['log_variances']

        print('TEST: Number infinite elements in variances: ', self.count_infinite_elements(test_dict['variances']))

        # convert torch tensor to numpy
        for key in test_dict.keys():
            test_dict[key] = test_dict[key].data.detach().cpu().numpy()

        metric_string = 'TEST:   '
        for metric in self.error_metrics:
            metric_string += ' {}: {:.3f},'.format(metric, test_metrics[metric])
        print(metric_string)
        return test_metrics, test_dict, metric_string

    def count_infinite_elements(self, x):
        return torch.sum(torch.logical_not(torch.isfinite(x))).item()
