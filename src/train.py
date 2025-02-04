import torch
import logging
import numpy as np
from models import UNet
from utils.parser import setup_parser
from pathlib import Path
from torchsummary import summary
import json
from trainer import Trainer
from utils.plots import plot_hist2d
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # set parameters / parse arguments
    parser = setup_parser()
    args, unknown = parser.parse_known_args()

    # log train/val metrics in tensorboard
    tensorboard_log_dir = Path(args.out_dir)/'log'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Setup model
    model = UNet(n_channels=args.num_ch, bilinear=args.bilinear) 
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    model.to(device)
    logging.info(f'Using device {device}')
    # Setup Trainer
    trainer = Trainer(model=model, log_dir=tensorboard_log_dir, args=args)

    print('TRAIN: ', len(trainer.ds_train))
    print('VAL:   ', len(trainer.ds_val))
    print('TEST:  ', len(trainer.ds_test))

    # train
    if os.path.exists(Path(args.out_dir) / 'best_weights.pt') or args.skip_training:
        print("MODEL WAS ALREADY TRAINED. SKIP TRAINING! (because the file 'best_weights.pt' exists already)")
    else:
        trainer.train()
        
        

    # --- test ---
    if os.path.exists(Path(args.out_dir) / 'confusion.png'):
        print("MODEL WAS ALREADY TESTED. SKIP TESTING! (because the file 'confusion.png' exists already)")
    else:
        test_metrics, test_dict, test_metric_string = trainer.test()

        # save results
        with open(Path(args.out_dir) / 'results.txt', 'w') as f:
            f.write(test_metric_string)

        with open(Path(args.out_dir) / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f)

        for key in test_dict.keys():
            np.save(file=Path(args.out_dir) / '{}.npy'.format(key), arr=test_dict[key])
        #np.save(file=Path(args.out_dir) / 'test_indices.npy', arr=trainer.test_indices)

        # plot confusion ground truth vs. prediction
        plot_hist2d(x=test_dict['targets'], y=test_dict['predictions'], ma=args.max_gt, step=args.max_gt / 10,
                    out_dir=args.out_dir, figsize=(8, 6))
