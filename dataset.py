# pyright: reportUnboundVariable=false
import os
import numpy as np
import torch
import torch.utils.data as data
import random
from utils import windowLevelNormalize
from scipy.ndimage import zoom, rotate

class SegDataset3D(data.Dataset):
    def __init__(self, available_ims, imagedir, maskdir, rotate_augment=False, scale_augment=False, one_hot_masks=True, test=False):
        self.available_im_fnames = available_ims
        self.imagedir = imagedir
        self.maskdir = maskdir
        self.one_hot_masks = one_hot_masks
        self.rotate_augment = rotate_augment
        self.scale_augment = scale_augment
        self.test = test
        if self.test:
            self.one_hot_masks = False

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        imageToUse = np.load(os.path.join(self.imagedir, self.available_im_fnames[idx]))
        maskToUse = np.load(os.path.join(self.maskdir, self.available_im_fnames[idx]))
        image, target_mask = self.generatePair(imageToUse, maskToUse)
        sample = {'image': image, 'target_mask': target_mask, 'fname': self.available_im_fnames[idx]}
        return sample

    def __len__(self):
        return len(self.available_im_fnames)

    def generatePair(self, image, mask):
        image = image.astype(float)        # float conversion here
        # data augmentations
        if self.test:
            pass
        else:
            if self.rotate_augment and random.random()<0.75:
                # decide the rotation axis 50:50 pitch : yaw
                if random.choice([True,False]):
                    pitch_angle = np.clip(np.random.normal(loc=0,scale=3), -10, 10)
                    image = self.rotation(image, pitch_angle, rotation_plane=(0,1), is_mask=False)
                    mask = self.rotation(mask, pitch_angle, rotation_plane=(0,1), is_mask=True)
                else:
                    yaw_angle = np.clip(np.random.normal(loc=0,scale=2.5), -7.5, 7.5)
                    image = self.rotation(image, yaw_angle, rotation_plane=(1,2), is_mask=False)
                    mask = self.rotation(mask, yaw_angle, rotation_plane=(1,2), is_mask=True)
            if self.scale_augment and random.random()<0.5:
                scale_factor = np.clip(np.random.normal(loc=1.0,scale=0.075), 0.8, 1.2)
                image = self.scale(image, scale_factor, is_mask=False)
                mask = self.scale(mask, scale_factor, is_mask=True)
        
        # Post-augmentations, add channels axis and normalise
        # After augmentations, perform window and level contrast normalisation (add the channels axis here)
        image = windowLevelNormalize(image[..., np.newaxis], level=1512, window=3024) # -> !!!! for CT !!!! (not sure what you're using)
        
        # convert to one-hot mask if required (e.g. for DICE loss)
        if self.one_hot_masks:
            mask = (np.arange(mask.max()+1) == mask[...,None]).astype(int)
        
        # check input data for problems
        assert not np.any(np.isnan(image))
        return image, mask

    def scale(self, image, scale_factor, is_mask):
        # scale the image or mask using scipy zoom function
        order = 0 if is_mask else 3
        height, width, depth = image.shape
        zheight = int(np.round(scale_factor*height))
        zwidth = int(np.round(scale_factor*width))
        zdepth = int(np.round(scale_factor*depth))
        # zoomed out
        if scale_factor < 1.0:
            new_image = np.zeros_like(image)
            ud_buffer = (height-zheight) // 2
            ap_buffer = (width-zwidth) // 2
            lr_buffer = (depth-zdepth) // 2
            new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth, lr_buffer:lr_buffer+zdepth] = zoom(input=image, zoom=scale_factor, order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
            return new_image
        elif scale_factor > 1.0:
            new_image = zoom(input=image, zoom=scale_factor, order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
            ud_extra = (new_image.shape[0] - height) // 2
            ap_extra = (new_image.shape[1] - width) // 2
            lr_extra = (new_image.shape[2] - depth) // 2
            new_image = new_image[ud_extra:ud_extra+height, ap_extra:ap_extra+width, lr_extra:lr_extra+depth]
            return new_image
        return image
    
    def rotation(self, image, rotation_angle, rotation_plane, is_mask):
        # rotate the image or mask using scipy rotate function
        order = 0 if is_mask else 3
        return rotate(input=image, angle=rotation_angle, axes=rotation_plane, reshape=False, order=order, mode='nearest')
