import argparse
import numpy as np
import os



def setup_parser():
    """
    Setup parser with default settings.
    Returns: argparse.ArgumentParser() object
    """

    parser = argparse.ArgumentParser()

    #### Result & Best Expriment here
    parser.add_argument("--out_dir", default='./tmp/', help="output directory for the experiment") 
    
    
    parser.add_argument("--all_checkpoint_dir", required = True, help="all_checkpoint directory")


    #
    parser.add_argument("--train_mode", default='scratch', help="Start training from sratch 'scratch' or continue'resume' (if continue please provide model weight)", choices=['scratch', 'resume'])

    # training params
    parser.add_argument("--skip_training", type=str2bool, nargs='?', const=True, default=False, help="do not optimize parameters (i.e. run test only)")
    parser.add_argument("--num_workers", default=os.cpu_count(), help="Number of workers for pytorch Dataloader", type=int)
    parser.add_argument("--loss_key", default='gaussian_nll', help="Loss keys", choices=['MSE', 'MAE', 'gaussian_nll', 'laplacian_nll'])
    parser.add_argument("--batch_size", default=32, help="batch size at train/val time. (number of samples per iteration)", type=int)
    parser.add_argument("--nb_epoch", default=100, help="number of epochs to train", type=int)
    parser.add_argument("--base_learning_rate", default=1e-7, help="initial learning rate", type=float)
    parser.add_argument("--l2_lambda", default=1e-8, help="L2 regularizer on weights hyperparameter", type=float)
    parser.add_argument("--optimizer", default='ADAM', help="optimizer name, choose 'ADAM' or 'SGD' ", choices=['ADAM', 'SGD'])
    parser.add_argument("--momentum", default=0.9, help="momentum for SGD ", type=float)
    parser.add_argument("--gradient_clipping", default=1.0, help="gradient clipping ", type=float)
    parser.add_argument('--amp', default=False, action='store_true' ,help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--num_ch', default=14, help='Number of input_channels' , type=int)
    parser.add_argument('--save_checkpoint', default=True, action='store_true' ,help='Save weight every epoch?')
    parser.add_argument('--wabdb_name', default='AGB_UNET', action='store_true' ,help='Wandb Project name')
     
    # Paramerter for filter in visulization step
    parser.add_argument("--min_gt", default=0, help="Filter target range: Keep samples >= min_gt", type=float)
    parser.add_argument("--max_gt", default=1000, help="Filter target range: Keep samples <= max_gt", type=float)
    parser.add_argument("--pearson_thresh", default=0.95, help="scalar (float) [0,1], quality criteria to filter data ", type=float)
    
    #dataset params
    parser.add_argument("--train_input_data_dir", required = True , help="Training input data directory with folder train/test/val.")
    parser.add_argument("--train_label_data_dir", required = True , help="Training label data directory with folder train/test/val.")
    
    # generalization experiment settings
    parser.add_argument("--model_weights_path", help="Pre-trained model weights path (e.g. best_weight.h5) ")

    return parser


# --- Helper functions to parse arguments ---

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v.lower() in ('none', '', 'nan', '0', '0.0'):
        return None
    else:
        return float(v)


def str_or_none(v):
    if v.lower() in ('none', '', 'nan', '0', '0.0'):
        return None
    else:
        return str(v)

class StoreAsArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(StoreAsArray, self).__call__(parser, namespace, values, option_string)
