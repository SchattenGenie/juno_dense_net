from tqdm import tqdm
import numpy as np
import torch


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.val = None
        self.avg = 0.
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0.

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1. - self.momentum)
        self.val = val


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class CustomDataLoader:
    def __init__(self, X, y, batch_size=100):
        self._X = X
        self._y = y
        self._batch_size = batch_size

    def __iter__(self):
        n = len(self._X)
        indicies = np.arange(n)
        np.random.shuffle(indicies)
        for idx in range(0, n, self._batch_size):
            yield self._X[indicies[idx:min(idx + self._batch_size, n)]], self._y[
                indicies[idx:min(idx + self._batch_size, n)]]
        return self

    def __len__(self):
        return len(self._X) // self._batch_size + 1

def perform_epoch(model, loader, loss_function, device, optimizer=None, epoch=None):
    """

    Performs one training or testing epoch, returns a tuple of mean loss and mean accuracy.\
    If 'optimizer' is not None, performs an optimization step.

    """
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    cum_loss = 0
    cum_batch_size = 0
    with NullContext() if is_train else torch.no_grad():
        for X, y in tqdm(loader):
            batch_size = X.shape[0]
            cum_batch_size += batch_size

            X = X
            y = y

            preds = model(X)
            loss = loss_function(preds, y, epoch=epoch)
            cum_loss += loss.item() * batch_size

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    mean_loss = cum_loss / cum_batch_size

    return mean_loss
