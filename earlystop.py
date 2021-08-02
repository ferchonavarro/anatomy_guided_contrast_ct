import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False,  mode='increasing'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode= mode
        self.val_loss_min = np.Inf




    def __call__(self, score, model, optimizer, epoch, name,):

        if self.mode=='increasing':
            if self.best_score is None:
                self.best_score = score
                auxname = name.split('.')[0]
                name = auxname + '_best.tar'
                self.save_checkpoint(score, model, optimizer, epoch, name)

            elif score >= self.best_score:
                self.best_score = score
                auxname = name.split('.')[0]
                name = auxname + '_best.tar'
                self.save_checkpoint(score, model, optimizer, epoch, name)
                self.counter = 0
            else:
                self.counter += 1
                # self.save_checkpoint(val_loss, model, optimizer, epoch, name)
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else: #decreasing
            if self.best_score is None:
                self.best_score = score
                auxname = name.split('.')[0]
                name = auxname + '_best.tar'
                self.save_checkpoint(score, model, optimizer, epoch, name)

            elif score <= self.best_score:
                self.best_score = score
                auxname = name.split('.')[0]
                name = auxname + '_best.tar'
                self.save_checkpoint(score, model, optimizer, epoch, name)
                self.counter = 0
            else:
                self.counter += 1
                # self.save_checkpoint(val_loss, model, optimizer, epoch, name)
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True




    def save_checkpoint(self, val_loss, model, optimizer, epoch, name):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'score': self.best_score,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss}, name)
        self.val_loss_min = val_loss
