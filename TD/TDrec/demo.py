'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''
import os
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import shutil
import numpy as np
from subprocess import call

import torch

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.data_augmentation import preprocess_img
from lib.solver import Solver
from lib.voxel import voxel2obj


DEFAULT_WEIGHTS = 'weight/checkpoint.80000.pth'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def load_demo_images(mode=0):
    img_h = cfg.CONST.IMG_H
    img_w = cfg.CONST.IMG_W
    
    imgs = []
    
    for i in range(9): #3
        img = Image.open('seg_out/%d.png' % (i + mode*9) )#imgs
        img = img.resize((img_h, img_w), Image.ANTIALIAS)
        img = preprocess_img(img, train=False)
        imgs.append([np.array(img).transpose( \
                        (2, 0, 1)).astype(np.float32)])
    ims_np = np.array(imgs).astype(np.float32)
    if(mode==0):
        print('Loading lower teeth images is done.')
    else:
        print('Loading upper teeth images is done.')
    return torch.from_numpy(ims_np)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    # load images
    print('Loading lower teeth images...')
    lower_teeth_imgs = load_demo_images(0)
    print('Loading upper teeth images...')
    upper_teeth_imgs = load_demo_images(1)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass()  # instantiate a network
    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    solver = Solver(net)                # instantiate a solver
    solver.load(DEFAULT_WEIGHTS)

    # Run the network
    print('Constructing lower teeth model ...')
    voxel_prediction, _ = solver.test_output(lower_teeth_imgs)
    voxel_prediction = voxel_prediction.detach().cpu().numpy()

    # Save the prediction to an OBJ file (mesh file).
    print('Saving lower teeth model ...')
    voxel2obj('3Dmodel/lower_teeth.obj', voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)

    # Run the network
    print('Constructing upper teeth model ...')
    voxel_prediction, _ = solver.test_output(upper_teeth_imgs)
    voxel_prediction = voxel_prediction.detach().cpu().numpy()

    # Save the prediction to an OBJ file (mesh file).
    print('Saving upper teeth model ...')
    voxel2obj('3Dmodel/upper_teeth.obj', voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    
    main()
    
