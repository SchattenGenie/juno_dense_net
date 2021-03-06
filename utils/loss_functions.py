import torch


class CombinatorialLoss:
    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __call__(self, y_true, y_pred, energy, epoch=None):
        mse_l = mse(y_true, y_pred, epoch=epoch)
        mae_l = mae(y_true, y_pred, epoch=epoch)
        energy_resolution_mse_l = energy_resolution_mse(y_true, y_pred, energy, epoch=epoch)
        energy_resolution_mae_l = energy_resolution_mae(y_true, y_pred, energy, epoch=epoch)
        return self._coeffs[0] * mse_l + self._coeffs[1] * mae_l + \
               self._coeffs[2] * energy_resolution_mse_l + self._coeffs[3] * energy_resolution_mae_l


def mse(y_true, y_pred, epoch=None):
    return (y_true.view(-1) - y_pred.view(-1)).pow(2).mean().sqrt()


def mae(y_true, y_pred, epoch=None):
    return (y_true.view(-1) - y_pred.view(-1)).abs().mean()


def energy_resolution_mse(y_true, y_pred, energy, epoch=None):
    # a small hack to stabilize training
    if epoch is None or epoch > 1:
        err = (y_true - y_pred) / energy.view(-1, 1)
        return err.pow(2).mean().sqrt()
    else:
        return mse(y_true, y_pred)


def energy_resolution_mae(y_true, y_pred, energy, epoch=None):
    # a small hack to stabilize training
    if epoch is None or epoch > 1:
        err = (y_true - y_pred) / energy.view(-1, 1)
        return err.abs().mean()
    else:
        return mae(y_true, y_pred)


def energy_resolution_mse_with_mse(y_true, y_pred, energy, epoch=None):
    # a small hack to stabilize training
    if epoch is None or epoch > 1:
        err = (y_true - y_pred) / energy.view(-1)
        return (mse(y_true, y_pred) + err.pow(2).abs().mean()) / 2.
    else:
        return mse(y_true, y_pred)


def energy_resolution_sqrt(y_true, y_pred, epoch=None):
    # a small hack to stabilize training
    if epoch is None or epoch > 1:
        err = (y_true.view(-1) - y_pred.view(-1)) / y_true.sqrt().view(-1)
        return err.pow(2).mean().sqrt()
    else:
        return mse(y_true, y_pred)


def energy_resolution_mse_shifted(y_true, y_pred, epoch=None, shift=0.5):
    # a small hack to stabilize training
    if epoch is None or epoch > 1:
        err = (y_true.view(-1) - y_pred.view(-1)) / (shift + y_true).view(-1)
        return err.pow(2).mean().sqrt()
    else:
        return mse(y_true, y_pred)


def energy_resolution_mean(y_true, y_pred, epoch=None):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.mean()


def energy_resolution_std(y_true, y_pred, epoch=None):
    err = (y_true.view(-1) - y_pred.view(-1)) / y_true.view(-1)
    return err.std()
