import torch


def mse(y_true, y_pred, epoch=None):
    return (y_true.view(-1) - y_pred.view(-1)).pow(2).mean().sqrt()


def mae(y_true, y_pred, epoch=None):
    return (y_true.view(-1) - y_pred.view(-1)).abs().mean()


def energy_resolution_mse(y_true, y_pred, epoch=None):
    # a small hack to stabilize training
    if epoch is None or epoch > 1:
        err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
        return err.pow(2).mean().sqrt()
    else:
        return mse(y_true, y_pred)

def energy_resolution_mean(y_true, y_pred, epoch=None):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.mean()


def energy_resolution_std(y_true, y_pred, epoch=None):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.std()
