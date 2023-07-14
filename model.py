import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResSegNet(nn.Module):
    def __init__(self, nClass, deep_supervision=True, ff=1, input_channels=1):
        super(ResSegNet, self).__init__()
        self.deep_supervision=deep_supervision
        # Channels-per-group
        c_p_g = 8
        ## Down Block 1
        self.db1_c1 = nn.Conv3d(in_channels=int(input_channels), out_channels=16*ff, kernel_size=(3,3,3), padding=1)
        self.db1_gn1 = nn.GroupNorm(num_groups=int((16*ff)/c_p_g), num_channels=16*ff)
        self.db1_c2 = nn.Conv3d(in_channels=16*ff, out_channels=32*ff, kernel_size=(3,3,3), padding=1)
        self.db1_gn2 = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
        self.db1_sc = nn.Conv3d(in_channels=int(input_channels), out_channels=32*ff, kernel_size=(1,1,1))
        self.db1_sgn = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
        ## Down Block 2
        self.db2_c1 = nn.Conv3d(in_channels=32*ff, out_channels=32*ff, kernel_size=(3,3,3), padding=1)
        self.db2_gn1 = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
        self.db2_c2 = nn.Conv3d(in_channels=32*ff, out_channels=64*ff, kernel_size=(3,3,3), padding=1)
        self.db2_gn2 = nn.GroupNorm(num_groups=int((16*ff)/c_p_g), num_channels=64*ff)
        self.db2_sc = nn.Conv3d(in_channels=32*ff, out_channels=64*ff, kernel_size=(1,1,1))
        self.db2_sgn = nn.GroupNorm(num_groups=int((16*ff)/c_p_g), num_channels=64*ff)
        ## Down Block 3
        self.db3_c1 = nn.Conv3d(in_channels=64*ff, out_channels=64*ff, kernel_size=(3,3,3), padding=1)
        self.db3_gn1 = nn.GroupNorm(num_groups=int((64*ff)/c_p_g), num_channels=64*ff)
        self.db3_c2 = nn.Conv3d(in_channels=64*ff, out_channels=128*ff, kernel_size=(3,3,3), padding=1)
        self.db3_gn2 = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        self.db3_sc = nn.Conv3d(in_channels=64*ff, out_channels=128*ff, kernel_size=(1,1,1))
        self.db3_sgn = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        ## Base Block 4
        self.bb4_c1 = nn.Conv3d(in_channels=128*ff, out_channels=128*ff, kernel_size=(3,3,3), padding=1)
        self.bb4_gn1 = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        self.bb4_c2 = nn.Conv3d(in_channels=128*ff, out_channels=256*ff, kernel_size=(3,3,3), padding=1)
        self.bb4_gn2 = nn.GroupNorm(num_groups=int((256*ff)/c_p_g), num_channels=256*ff)
        self.bb4_sc = nn.Conv3d(in_channels=128*ff, out_channels=256*ff, kernel_size=(1,1,1))
        self.bb4_sgn = nn.GroupNorm(num_groups=int((256*ff)/c_p_g), num_channels=256*ff)
        # transpose convolution
        self.bb4_tc = nn.ConvTranspose3d(in_channels=256*ff, out_channels=256*ff, kernel_size=(2,2,2), stride=(2,2,2))
        self.bb4_gn3 = nn.GroupNorm(num_groups=int((256*ff)/c_p_g), num_channels=256*ff)
        ## Up Block 3
        self.ub3_c1 = nn.Conv3d(in_channels=(128+256)*ff, out_channels=128*ff, kernel_size=(3,3,3), padding=1)
        self.ub3_gn1 = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        self.ub3_c2 = nn.Conv3d(in_channels=128*ff, out_channels=128*ff, kernel_size=(3,3,3), padding=1)
        self.ub3_gn2 = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        self.ub3_sc = nn.Conv3d(in_channels=(128+256)*ff, out_channels=128*ff, kernel_size=(1,1,1))
        self.ub3_sgn = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        # transpose convolution
        self.ub3_tc = nn.ConvTranspose3d(in_channels=128*ff, out_channels=128*ff, kernel_size=(2,2,2), stride=(2,2,2))
        self.ub3_gn3 = nn.GroupNorm(num_groups=int((128*ff)/c_p_g), num_channels=128*ff)
        ## Up Block 2
        self.ub2_c1 = nn.Conv3d(in_channels=(64+128)*ff, out_channels=64*ff, kernel_size=(3,3,3), padding=1)
        self.ub2_gn1 = nn.GroupNorm(num_groups=int((64*ff)/c_p_g), num_channels=64*ff)
        self.ub2_c2 = nn.Conv3d(in_channels=64*ff, out_channels=64*ff, kernel_size=(3,3,3), padding=1)
        self.ub2_gn2 = nn.GroupNorm(num_groups=int((64*ff)/c_p_g), num_channels=64*ff)
        self.ub2_sc = nn.Conv3d(in_channels=(64+128)*ff, out_channels=64*ff, kernel_size=(1,1,1))
        self.ub2_sgn = nn.GroupNorm(num_groups=int((64*ff)/c_p_g), num_channels=64*ff)
        # transpose convolution
        self.ub2_tc = nn.ConvTranspose3d(in_channels=64*ff, out_channels=64*ff, kernel_size=(2,2,2), stride=(2,2,2))
        self.ub2_gn3 = nn.GroupNorm(num_groups=int((64*ff)/c_p_g), num_channels=64*ff)
        ## Up Block 1
        self.ub1_c1 = nn.Conv3d(in_channels=(32+64)*ff, out_channels=32*ff, kernel_size=(3,3,3), padding=1)
        self.ub1_gn1 = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
        self.ub1_c2 = nn.Conv3d(in_channels=32*ff, out_channels=32*ff, kernel_size=(3,3,3), padding=1)
        self.ub1_gn2 = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
        self.ub1_sc = nn.Conv3d(in_channels=(32+64)*ff, out_channels=32*ff, kernel_size=(1,1,1))
        self.ub1_sgn = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
        ## Output Convolution
        if self.deep_supervision:
            # deep supervision bottleneck convolutions
            self.deep4_bottleneck = nn.Conv3d(in_channels=256*ff, out_channels=32*ff, kernel_size=(1,1,1))
            self.bottle4_gn = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
            self.deep3_bottleneck = nn.Conv3d(in_channels=128*ff, out_channels=32*ff, kernel_size=(1,1,1))
            self.bottle3_gn = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
            self.deep2_bottleneck = nn.Conv3d(in_channels=64*ff, out_channels=32*ff, kernel_size=(1,1,1))
            self.bottle2_gn = nn.GroupNorm(num_groups=int((32*ff)/c_p_g), num_channels=32*ff)
            self.out_conv = nn.Conv3d(in_channels=32*ff*4, out_channels=nClass, kernel_size=(1,1,1))
        else:
            self.out_conv = nn.Conv3d(in_channels=32*ff, out_channels=nClass, kernel_size=(1,1,1))
        self.out_gn = nn.GroupNorm(num_groups=1, num_channels=nClass)   # currently layer norm for the output
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        # DB1
        down1 = F.relu(self.db1_sgn(self.db1_sc(x)))    # save for residual skip
        x = F.relu(self.db1_gn1(self.db1_c1(x)))
        x = F.relu(self.db1_gn2(self.db1_c2(x)))        
        down1 = down1 + x                               # save for UNet skip
        x = F.max_pool3d(down1, (2,2,2))
        # DB2
        down2 = F.relu(self.db2_sgn(self.db2_sc(x)))
        x = F.relu(self.db2_gn1(self.db2_c1(x)))
        x = F.relu(self.db2_gn2(self.db2_c2(x)))
        down2 = down2 + x
        x = F.max_pool3d(down2, (2,2,2))
        # DB3
        down3 = F.relu(self.db3_sgn(self.db3_sc(x)))
        x = F.relu(self.db3_gn1(self.db3_c1(x)))
        x = F.relu(self.db3_gn2(self.db3_c2(x)))
        down3 = down3 + x
        x = F.max_pool3d(down3, (2,2,2))
        # BB4
        res_skip4 = F.relu(self.bb4_sgn(self.bb4_sc(x)))
        x = F.relu(self.bb4_gn1(self.bb4_c1(x)))
        x = F.relu(self.bb4_gn2(self.bb4_c2(x)))
        x = x + res_skip4
        deep4 = F.relu(self.bb4_gn3(self.bb4_tc(x)))
        # UB3
        x = torch.cat((deep4, down3), dim=1)
        res_skip3 = F.relu(self.ub3_sgn(self.ub3_sc(x)))
        x = F.relu(self.ub3_gn1(self.ub3_c1(x)))
        x = F.relu(self.ub3_gn2(self.ub3_c2(x)))
        x = x + res_skip3
        deep3 = F.relu(self.ub3_gn3(self.ub3_tc(x)))
        # UB2
        x = torch.cat((deep3, down2), dim=1)
        res_skip2 = F.relu(self.ub2_sgn(self.ub2_sc(x)))
        x = F.relu(self.ub2_gn1(self.ub2_c1(x)))
        x = F.relu(self.ub2_gn2(self.ub2_c2(x)))
        x = x + res_skip2
        deep2 = F.relu(self.ub2_gn3(self.ub2_tc(x)))
        # UB1
        x = torch.cat((deep2, down1), dim=1)
        res_skip1 = F.relu(self.ub1_sgn(self.ub1_sc(x)))
        x = F.relu(self.ub1_gn1(self.ub1_c1(x)))
        x = F.relu(self.ub1_gn2(self.ub1_c2(x)))
        x = x + res_skip1
        # Output Conv
        if self.deep_supervision:
            deep4 = F.interpolate(F.relu(self.bottle4_gn(self.deep4_bottleneck(deep4))), scale_factor=4, mode='trilinear' , align_corners=False)
            deep3 = F.interpolate(F.relu(self.bottle3_gn(self.deep3_bottleneck(deep3))), scale_factor=2, mode='trilinear' , align_corners=False)
            deep2 = F.relu(self.bottle2_gn(self.deep2_bottleneck(deep2)))
            x = torch.cat((x, deep2, deep3, deep4), dim=1)
        x = self.out_gn(self.out_conv(x))
        return x

    def set_all_to_train(self, logger=None):
        for layer in self.children():
            for param in layer.parameters():
                param.requires_grad = True
        if logger:
            logger.info("Unlocked all layers")
        else:
            print("Unlocked all layers")

    def lock_layers(self, logger=None):
        for layer in self.children():
            for param in layer.parameters():
                param.requires_grad = False
        if logger:
            logger.info("Locked all layers")
        else:
            print("Locked all layers")

    def load_best(self, checkpoint_dir, logger=None, for_transfer=False):
        # load previous best weights --> prevent using a previous bad state as starting point for fine tuning
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            if for_transfer:
                new_key = new_key.replace('out', 'edward') # Don't load the output convolution weights
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, _ in list(renamed_dict.items())])
        else:
            return

    def load_path(self, path):
        # load previous best weights --> prevent using a previous bad state as starting point for fine tuning
        model_dict = self.state_dict()
        state = torch.load(path)
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        print("Loaded layers from previous best checkpoint:")
        print([k for k, _ in list(renamed_dict.items())])