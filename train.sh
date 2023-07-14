#!/bin/bash
python3 train.py --image_dir "/home/ed/segmentation_work/miccai_2015_data/headHunted/CTs/"\
                 --mask_dir "/home/ed/segmentation_work/miccai_2015_data/headHunted/Structures/"\
                 --output_dir "./test_out/"\
                 --fold_num 2
