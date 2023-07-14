import argparse as ap
import numpy as np
from os.path import join

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from trainer import SegNetTrainer, exp_log_loss
from model import ResSegNet
from utils import get_logger, get_number_of_learnable_parameters, str2bool, k_fold_split_train_val_test, getFiles
from dataset import SegDataset3D

## Add argparse
def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D Segmentation with the deepmind data")
    parser.add_argument("--image_dir", type=str, help="Path to the directory containing the images")
    parser.add_argument("--mask_dir", type=str, help="Path to the directory containing the masks")
    parser.add_argument("--output_dir", type=str, help="Path to the directory to save the model checkpoints and logs")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5,0], default=1, type=int, help="The fold number for the kfold cross validation")
    global args
    args = parser.parse_args()

def main():
    # get args
    setup_argparse()
    global args
    image_dir = args.image_dir
    mask_dir = args.mask_dir

    # choose the images to use in this training fold
    dataset_size = len(getFiles(image_dir))
    train_inds, val_inds, _ = k_fold_split_train_val_test(fold_num=args.fold_num, dataset_size=dataset_size, seed=2305)

    train_fnames = [sorted(getFiles(image_dir))[i] for i in train_inds]
    val_fnames = [sorted(getFiles(image_dir))[i] for i in val_inds]

    checkpoint_dir = join(args.output_dir, f"fold{args.fold_num}/")

    # Create main logger
    logger = get_logger('SegNetTrainer')

    # Set oar indices
    nClass = 6      # nClass:   0 - Background
                    #           1 - w
                    #           2 - x
                    #           3 - y
                    #           4 - z
                    #           5 - z2

    # setup loss function      
    ''' 
    label_freq below should be a numpy array of length nClass where the values 
    represent the relative frequency of the segmentation classes. This helps
    large class imbalances caused by small structures.
    --> see get_freq function in utils.py
    '''
    label_freq = np.array([1, 1, 1, 1, 1, 1])
    loss_fn = exp_log_loss(label_freq=label_freq)

    # Create the model
    model = ResSegNet(nClass=nClass)

    # put the model on GPU
    device='cuda'
    model.to(device)

    # Create data loaders
    iters_to_accumulate = 4
    batch_size = 1
    train_data = SegDataset3D(available_ims=train_fnames, imagedir=image_dir, maskdir=mask_dir, scale_augment=True, rotate_augment=True, one_hot_masks=True)
    val_data = SegDataset3D(available_ims=val_fnames, imagedir=image_dir, maskdir=mask_dir, one_hot_masks=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Open all the layers to train
    model.set_all_to_train(logger)

    # Create the optimizer - only pass the parameters that require tuning for speed
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True)
    early_stop_patience = 64

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create model trainer
    trainer = SegNetTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=train_loader, val_loader=val_loader,
                            logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=200, patience=early_stop_patience, loss_fn=loss_fn, iters_to_accumulate=iters_to_accumulate)
    
    # Start training
    trainer.fit()

    # finally regenerate validation images for the best version of the model
    trainer._gen_best_val_images()
    
    # Romeo Dunn
    return

if __name__ == '__main__':
    main()