import numpy as np
import os
import time
import argparse as ap
from os.path import join

import torch
from torch.utils.data import DataLoader

from model import ResSegNet
from utils import getFiles, k_fold_split_train_val_test, str2bool
from dataset import SegDataset3D
import deepmind_metrics
from scipy import ndimage

## Add argparse
def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D Segmentation with the deepmind data")
    parser.add_argument("--image_dir", type=str, help="Path to the directory containing the images")
    parser.add_argument("--mask_dir", type=str, help="Path to the directory containing the masks")
    parser.add_argument("--model_dir", type=str, help="Path to the directory containing the model checkpoints and logs")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--deep_supervision", default=True, type=lambda x:bool(str2bool(x)), help="Use deep 3D UNet supervision?")
    global args
    args = parser.parse_args()

def main():
    # get args
    setup_argparse()
    global args
    
    image_dir = args.image_dir
    mask_dir = args.mask_dir

    # Set oar indices
    # default
    nClass = 6      # nClass:   0 - Background
                    #           1 - w
                    #           2 - x
                    #           3 - y
                    #           4 - z
                    #           5 - z2

    # Create the model
    model = ResSegNet(nClass=nClass, deep_supervision=args.deep_supervision)

    # put the model on GPU
    device='cuda'
    model.to(device)

    # prepare the model
    model.load_best(checkpoint_dir)
    model.lock_layers()
    model.eval()

    # choose the images to use in this training fold
    dataset_size = len(getFiles(image_dir))
    _, _, test_inds = k_fold_split_train_val_test(fold_num=args.fold_num, dataset_size=dataset_size, seed=2305)

    test_fnames = [getFiles(image_dir)[i] for i in test_inds]

    checkpoint_dir = join(args.model_dir, f"fold{args.fold_num}/")

    # Create data loaders
    test_data = SegDataset3D(available_ims=test_fnames, imagedir=image_dir, maskdir=mask_dir)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
    
    # create output directory
    segs_dir = join(checkpoint_dir, 'test_segs')
    out_dir = join(checkpoint_dir, "results")
    os.makedirs(segs_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # run the testing
    res = np.zeros(shape=(len(test_fnames), nClass-1, 2))
    for pat_idx, sample in enumerate(test_loader):
        image = sample['image']
        target_mask = sample['target_mask']
        fname = sample['fname'][0]
        # send tensors to the gpu
        image = image.type(torch.float).to(device)
        # inference
        image = image.permute(0,4,1,2,3)
        t = time.time()
        output = model(image)
        print(f"segmentation took: {time.time() - t:.2f} seconds")
        # convert to masks
        output = torch.squeeze(output)
        target_mask = torch.squeeze(target_mask)
        output = torch.argmax(output, dim=0)
        # convert to numpy
        output = output.cpu().numpy().astype(int)
        target_mask = target_mask.cpu().numpy().astype(int)
        # save output
        np.save(os.path.join(segs_dir, 'pred_exp_'+fname), output)
        np.save(os.path.join(segs_dir, 'gold_exp_'+fname), target_mask)
        # use deepmind metrics here
        '''
        You'll need to modify this bit with the spacing of the input images
        '''
        spacing = [2.5, 1, 1] # this bit
        # Background (0) excluded:
        first_oar_idx = 1
        for organ_idx, organ_num in enumerate(range(first_oar_idx, target_mask.max()+1)):
            # Need to binarise the masks for the metric computation
            gs = np.zeros(shape=target_mask.shape)
            pred = np.zeros(shape=output.shape)
            gs[(target_mask==organ_num)] = 1
            pred[(output==organ_num)] = 1
            # post-processing using scipy.ndimage.label to eliminate extraneous voxels
            labels, num_features = ndimage.label(input=pred, structure=np.ones((3,3,3)))
            sizes = ndimage.sum(pred, labels, range(num_features+1))
            pred[(labels!=np.argmax(sizes))] = 0
            # compute the surface distances
            surface_distances = deepmind_metrics.compute_surface_distances(gs.astype(bool), pred.astype(bool), spacing)
            # compute desired metric
            hausdorff95 = deepmind_metrics.compute_robust_hausdorff(surface_distances, percent=95.)
            meanDTA = deepmind_metrics.compute_average_surface_distance(surface_distances)
            # store result
            res[pat_idx, organ_idx, 0] = hausdorff95
            res[pat_idx, organ_idx, 1] = meanDTA

    np.save(os.path.join(out_dir, f'fold{args.fold_num}_results_vec.npy'), res)
    return

if __name__ == '__main__':
    main()