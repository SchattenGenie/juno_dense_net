import torch


def mse(y_true, y_pred):
    return (y_true.view(-1) - y_pred.view(-1)).pow(2).mean().sqrt()


def mae(y_true, y_pred):
    return (y_true.view(-1) - y_pred.view(-1)).abs().mean()


def energy_resolution_mse(y_true, y_pred):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.pow(2).mean().sqrt()


def energy_resolution_mean(y_true, y_pred):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.mean()


def energy_resolution_std(y_true, y_pred):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.std()
