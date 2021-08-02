from __future__ import print_function, division
from data_loader import ClassificationHandler
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score
eps = 1e-10
import logging
import shutil
from earlystop import EarlyStopping
from datetime import datetime
import json
from sklearn.metrics import roc_curve, auc, f1_score
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, RandFlip,RandRotate, RandZoom, ToTensor, ResizeWithPadOrCrop

class Trainer(object):
    """
    Trains a multilabel classification:
    example==> given a CT slice with segmentation labels tell which organs are present in CT slice
    used to train and validate

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param verification_batch_size: size of verification batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """

    def __init__(self, data_files, model, cost_function, optimizer, schedulers, stop_epoch=100,
                 target_names=None, labels=None, logging=False, opt_kwargs={}):
        self.model = model
        self.data_files = data_files
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.schedulers = schedulers
        self.opt_kwargs = opt_kwargs
        self.best_metric = 0
        self.best_model = ''
        self.epoch = 0
        self.global_step = 0
        self.target_names = target_names
        self.labels = labels
        self.stop_epoch = stop_epoch
        self.logging = logging

    def _initialize(self):

        if not self.restore:
            logging.info("Removing '{:}'".format(self.log_dir))
            shutil.rmtree(self.log_dir, ignore_errors=True)

        if not os.path.exists(self.log_dir):
            logging.info("Allocating '{:}'".format(self.log_dir))
            os.makedirs(self.log_dir)

        self.train_folder = os.path.join(self.log_dir, 'train')
        if not os.path.exists(self.train_folder):
            logging.info("Allocating '{:}'".format(self.train_folder))
            os.makedirs(self.train_folder)

        self.val_folder = os.path.join(self.log_dir, 'val')
        if not os.path.exists(self.val_folder):
            logging.info("Allocating '{:}'".format(self.val_folder))
            os.makedirs(self.val_folder)

        self.model_folder = os.path.join(self.log_dir, 'models')
        if not os.path.exists(self.model_folder):
            logging.info("Allocating '{:}'".format(self.model_folder))
            os.makedirs(self.model_folder)

        self.pred_folder = os.path.join(self.log_dir, 'predictions')
        if not os.path.exists(self.pred_folder):
            logging.info("Allocating '{:}'".format(self.pred_folder))
            os.makedirs(self.pred_folder)

        self.cm_folder = os.path.join(self.log_dir, 'cm')
        if not os.path.exists(self.cm_folder):
            logging.info("Allocating '{:}'".format(self.cm_folder))
            os.makedirs(self.cm_folder)

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(patience=self.stop_epoch, verbose=True, mode='increasing')


    def train(self, num_epochs, log_dir, eval_rate=1, display_rate=20, num_images=6, restore_file=None, batch_size=100,
              image_size=[224,224], num_classes=21, device='cuda:1'):

        self.since = datetime.now()
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.eval_rate = eval_rate
        self.display_rate = display_rate
        self.num_images = num_images
        self.num_classes = num_classes
        self.restore_file = restore_file
        self.image_size = image_size

        if restore_file is not None:
            self.restore=True
        else:
            self.restore=False

        self._initialize()

        if self.logging:
            self.train_logger = SummaryWriter(self.train_folder)
            self.val_logger = SummaryWriter(self.val_folder)

        if batch_size < self.num_images:
            self.num_images = batch_size


        ## DATA AUGMENTATIONS

        train_transforms = Compose([
            RandFlip(spatial_axis=(0,1), prob=0.5),
            RandRotate(range_x=(-180,180), prob=0.5, keep_size=True, padding_mode='zeros'),
            RandZoom(min_zoom=0.6, max_zoom=1.1, prob=0.5, keep_size=True),
            ResizeWithPadOrCrop(spatial_size=self.image_size,mode='constant'),
            ToTensor()])

        val_transforms = Compose(
            [ResizeWithPadOrCrop(spatial_size=self.image_size, mode='constant'),
             ToTensor()])

        data_handler1 = ClassificationHandler(files=self.data_files['train'],
                                              transform=train_transforms, Inference=False)

        data_provider1 = DataLoader(data_handler1, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

        data_handler2 = ClassificationHandler(files=self.data_files['val'],
                                              transform=val_transforms, Inference=True)
        data_provider2 = DataLoader(data_handler2, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False)

        self.dataloaders = {}
        self.dataloaders['train'] = data_provider1
        self.dataloaders['val'] = data_provider2
        self.dataset_sizes = {}
        self.dataset_sizes['train'] = len(data_handler1)
        self.dataset_sizes['val'] = len(data_handler2)
        self.device = torch.device(device)


        self.end_training = False
        self.current_iteration = 0

        if restore_file is not None:
            self.model.load_model(self.restore_file)

        for epoch in range(self.num_epochs):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            print('-' * 20)
            phase = 'train'
            print("<<<= Phase: %s =>>>" % phase)
            self.train_model(phase, epoch)
            if epoch % self.eval_rate == 0:
                phase = 'val'
                print("<<<= Phase: %s =>>>" % phase)
                self.evaluate_model(phase, epoch)
            ## logging epoch
            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            if self.end_training:
                break

        time_elapsed = datetime.now() - self.since
        print('Training complete in {}'.format(time_elapsed))
        print('Best val Acc: {:4f}'.format(self.best_metric))
        print('Best model: {}'.format(self.best_model))

    def train_model(self, phase, epoch):
        sum_loss_train = []
        gt=[]
        preds=[]
        probas= []
        self.model.train()
        for i_batch, sample_batched in enumerate(self.dataloaders[phase]):
            x = sample_batched[0].type(torch.FloatTensor)
            y = sample_batched[1].type(torch.LongTensor)
            logits = self.model(x.cuda(self.device))
            loss = self.cost_function(logits, y.cuda(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.current_iteration += 1
            sum_loss_train.append(loss.item())
            with torch.no_grad():
                _, predicted = torch.max(torch.softmax(logits, dim=1), 1)

                if x.shape[0]==1:
                    gt.extend(np.expand_dims(y.squeeze().numpy(), axis=0))
                    predaux = predicted.cpu().squeeze().detach().numpy()
                    preds.extend(np.expand_dims(predaux, axis=0))
                    probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
                else:
                    gt.extend(y.squeeze().numpy())
                    predaux = predicted.cpu().squeeze().detach().numpy()
                    preds.extend(predaux)
                    probas.extend(np.squeeze(torch.softmax(logits, dim=1).cpu().numpy()))

                gt.extend(y.numpy())
                preds.extend(predicted.cpu().squeeze().detach().numpy())
                acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
                auc = self.compute_auc_multi_class(y.numpy(), torch.softmax(logits, dim=1).cpu().numpy())
                f1score= f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

                if i_batch % self.display_rate == 0:
                    print('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss.item()) +' ]')
                    self.train_logger.add_scalar('loss', loss, self.current_iteration)
                    self.train_logger.add_scalar('acc', acc, self.current_iteration)
                    self.train_logger.add_scalar('auc', auc, self.current_iteration)
                    self.train_logger.add_scalar('f1score', f1score, self.current_iteration)
                if self.logging and i_batch == 0:
                    self.train_logger.add_images('images',np.expand_dims(x[0:self.num_images, 0, :, :].numpy(),axis=1),
                                                    self.current_iteration)

                del x, y, loss, logits, predicted
                torch.cuda.empty_cache()

        epoch_loss = np.mean(sum_loss_train)
        probas= np.asarray(probas)
        epoch_acc = accuracy_score(np.asarray(gt), np.asarray(preds))
        epoch_auc = self.compute_auc_multi_class(np.asarray(gt), probas)
        epoch_f1score = f1_score(np.asarray(gt), np.asarray(preds), average='macro')

        if self.logging:
            self.train_logger.add_scalar('epoch_loss', epoch_loss, epoch)
            self.train_logger.add_scalar('epoch_f1score', epoch_f1score, epoch)
            self.train_logger.add_scalar('epoch_acc', epoch_acc, epoch)
            self.train_logger.add_scalar('epoch_auc', epoch_auc, epoch)

        print('Phase: {}. Epoch {} Loss: {:.4f},  acc : {:.4f},  auc : {:.4f},  f1score : {:.4f}'.format(
            phase,epoch, epoch_loss, epoch_acc,epoch_auc, epoch_f1score))


    def evaluate_model(self, phase, epoch):
        self.model.eval()  # Set model to evaluate mode
        prefix = 'model'
        gt = []
        preds = []
        sum_loss_train = []
        probas =[]
        for i_batch, sample_batched in enumerate(self.dataloaders[phase]):
            x = sample_batched[0].type(torch.FloatTensor)
            y = sample_batched[1].type(torch.LongTensor)
            logits = self.model(x.cuda(self.device))
            with torch.no_grad():
                _, predicted = torch.max(torch.softmax(logits, dim=1), 1)

                if x.shape[0]==1:
                    gt.extend(np.expand_dims(y.squeeze().numpy(), axis=0))
                    predaux = predicted.cpu().squeeze().detach().numpy()
                    preds.extend(np.expand_dims(predaux, axis=0))
                    probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
                else:
                    gt.extend(y.squeeze().numpy())
                    predaux = predicted.cpu().squeeze().detach().numpy()
                    preds.extend(predaux)
                    probas.extend(np.squeeze(torch.softmax(logits, dim=1).cpu().numpy()))
                loss = self.cost_function(logits, y.cuda(self.device))
                sum_loss_train.append(loss.item())


                acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())

                auc = self.compute_auc_multi_class(y.numpy(), torch.softmax(logits, dim=1).cpu().numpy())

                f1score = f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

                if i_batch % self.display_rate == 0:
                    print('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss.item()) + '  acc -> ' + str(
                        acc) + '  auc -> ' + str(auc) + ' f1score -> ' + str(f1score))
                    self.val_logger.add_scalar('loss', loss, self.current_iteration)
                    self.val_logger.add_scalar('acc', acc, self.current_iteration)
                    self.val_logger.add_scalar('auc', auc, self.current_iteration)
                    self.val_logger.add_scalar('f1score', f1score, self.current_iteration)

                if self.logging and i_batch == 0:
                    self.val_logger.add_images('images', np.expand_dims(x[0:self.num_images, 0, :, :].numpy(),axis=1),
                                                    self.current_iteration)

                del x, y, loss, logits, predicted
                torch.cuda.empty_cache()

        epoch_loss = np.mean(sum_loss_train)

        self.schedulers.step(epoch_loss)
        probas = np.asarray(probas)
        epoch_acc = accuracy_score(np.asarray(gt), np.asarray(preds))
        print('Shape gt',np.shape(np.asarray(gt)))
        print('Shape probas', np.shape(probas))
        epoch_auc = self.compute_auc_multi_class(np.asarray(gt), probas)
        epoch_f1score = f1_score(np.asarray(gt), np.asarray(preds), average='macro')

        if logging:
            self.val_logger.add_scalar('epoch_loss', epoch_loss, epoch)
            self.val_logger.add_scalar('epoch_f1score', epoch_f1score, epoch)
            self.val_logger.add_scalar('epoch_acc', epoch_acc, epoch)
            self.val_logger.add_scalar('epoch_auc', epoch_auc, epoch)

        print('Phase: {}. Epoch {} Loss: {:.4f},  acc : {:.4f},  auc : {:.4f},  f1score : {:.4f}'.format(
            phase, epoch, epoch_loss, epoch_acc, epoch_auc, epoch_f1score))


        epochname = str(epoch).zfill(3)
        auxaux = prefix + '_epoch_' + str(epochname) + '.tar'
        aux = prefix + '.tar'

        filename = os.path.join(self.model_folder, aux)

        if epoch_f1score >= self.best_metric:
            self.best_metric = epoch_f1score
            self.best_model = auxaux

        self.early_stopping(epoch_f1score, self.model, self.optimizer, epoch, filename)
        if self.early_stopping.early_stop:
            self.end_training = True
            print("Early stopping")
            print(self.best_model)

    def compute_auc_multi_class(self, y_test, y_predict_proba):
        aucs=[]
        for i in range(self.num_classes):
            y_test_i = np.zeros(shape=y_test.shape, dtype=y_test.dtype)
            for index, x in enumerate(y_test):
                if y_test[index] ==i:
                    y_test_i[index]=1
                else:
                    y_test_i[index]=0
            fpr, tpr, _ = roc_curve(y_test_i, y_predict_proba[:, i])
            aux = auc(fpr, tpr)
            if np.isnan(aux):
                aux=0
            aucs.append(aux)

        return np.mean(aucs)







