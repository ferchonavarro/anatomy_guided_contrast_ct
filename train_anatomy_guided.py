from __future__ import print_function, division
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.optim import lr_scheduler
import sys
import os
import glob
local= True
if local:
    sys.path.append("/home/fercho/projects/neurorad")
else:
    sys.path.append("/home/navarrof/projects/neurorad")

from trainer import Trainer
from torchvision import models
import torch.nn as nn
import torch
from losses import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from itertools import combinations
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_folder', help='Path to .npz files folders',default='./npz_files',type=str)
    parser.add_argument(
        '--logging_folder',
        help='Path to folder to save training checkpoints', default='./log',type=str)
    parser.add_argument(
        '--batch_size', help='batch size for training',default=100,type=int)
    parser.add_argument(
        '--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument(
        '--num_epochs', help='epochs to train', default=100, type=int)
    parser.add_argument(
        '--num_classes', help='number of classes for classification', default=3, type=int)
    parser.add_argument(
        '--patience', help='number of epochs patience for early stopping', default=10, type=int)
    parser.add_argument(
        '--gpu', help='GPU number', default=1, type=int)

    args = parser.parse_args()

    """
    Input data and paths
    """

    # prefix = 'densenet169_single_slice_'
    folders = glob.glob(args.data_folder + '/*')
    folders.sort()

    yaux = np.zeros(len(folders))
    yaux[len(folders)//3:2*len(folders)//3] = 1
    yaux[2*len(folders)//3:] = 2


    # ### split the training data into train and test 20% test, 80% train
    random_state = 42
    # print(np.unique(versey, return_counts=True))
    x_train, x_aux, y_train, y_aux = train_test_split(folders, yaux, test_size=0.2, random_state=random_state)



    all_folders = x_train
    labels= list(y_train.astype('int'))


    print('='*50)
    print('Log folder')
    print(args.logging_folder)

    print('CT data folder')
    print(args.data_folder)


    print('Number of patients')
    print(len(all_folders))
    print('='*50)


    n_folds = 3
    rskf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

    indices = [15, 16, 17, 18, 19, 20, 21]
    modality = ['art', 'nat', 'pv']
    vertebrae = ['T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2']

    all_comb = []

    stuff = [7]
    cont = 0
    for L in stuff:
        for subset in combinations(vertebrae, L):
            all_comb.append(subset)
            print(cont + 1)
            print(subset)
            cont += 1

    for fold, (train_index, val_index) in enumerate(rskf.split(all_folders, labels)):
        combination= all_comb[0]
        name = ''
        for n in combination:
            name = name + '_' + str(n)

        foldname = str(fold + 1).zfill(3)
        master_folder = os.path.join(args.logging_folder, 'fold_{}'.format(foldname)+name)
        if not os.path.exists(master_folder):
            os.makedirs(master_folder)

        print('Model Name', master_folder)

        train_files = []

        for n in combination:
            for i in train_index:
                folder = all_folders[i]
                current_study = '/*{}*.npz'.format(n)
                files = glob.glob(folder + current_study)
                train_files.extend(files)

        val_files = []
        for n in combination:
            for i in val_index:
                folder = all_folders[i]
                current_study = '/*{}*.npz'.format(n)
                files = glob.glob(folder + current_study)
                val_files.extend(files)


        """
        DataLoaders and Logs
        """
        mode = 'allfeatures'
        data_files = {}
        data_files['train'] = train_files
        data_files['val'] = val_files



        ########################## Network
        model = models.densenet169(pretrained=True)
        model.classifier = nn.Linear(1664, args.num_classes)
        size = (224, 224)

        cuda_device = "cuda:0"
        device = torch.device(cuda_device)

        model_ft = model.to(device)
        loss = CrossEntropyLoss()

        loss = loss.to(device)

        """
        Optimizer and schedulers
        """
        optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=args.lr)
        decay_step = 10
        gamma = 0.1
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.num_epochs, eta_min=1e-7)
        trainer_instance = Trainer(data_files=data_files,
                                   model=model_ft,
                                   cost_function=loss,
                                   optimizer=optimizer_ft,
                                   schedulers=exp_lr_scheduler,
                                   stop_epoch=args.patience,
                                   target_names=None, logging=True)

        trainer_instance.train(num_epochs=args.num_epochs,
                               log_dir=master_folder,
                               eval_rate=1,
                               display_rate=20,
                               num_images=3,
                               restore_file=None,
                               batch_size=args.batch_size,
                               image_size=size,
                               device=cuda_device,
                               num_classes=args.num_classes)
