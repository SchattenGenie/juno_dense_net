import os
import json
import numpy as np
from comet_ml import Experiment as CometExperiment
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
sns.set(context='paper', style="whitegrid", font_scale=1.5)
LOG_HIST = False


def warn():
    import traceback
    import warnings
    warnings.warn(traceback.format_exc())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Logger(object):
    @profile
    def log_metrics(self, net, loader, name: str, device: str):
        y_true = []
        y_pred = []

        with torch.no_grad():
            if isinstance(loader, tuple):
                X_test, y_true = loader
                y_true = y_true.view(-1).to(device)
                y_pred = net(X_test.to(device)).view(-1)
            else:
                for X, y in tqdm(loader):
                    X = X.to(device)
                    y = y.to(device)
                    preds = net(X)
                    y_pred.append(preds.cpu().view(-1))
                    y_true.append(y.cpu().view(-1))
                y_pred = torch.cat(y_pred).view(-1)
                y_true = torch.cat(y_true).view(-1)

        metrics = {}
        figures = {}
        metrics["mse"] = self._mse(y_true, y_pred)
        metrics["mae"] = self._mae(y_true, y_pred)
        mean_er, std_er, f = self._energy_resolution_gaussian_fit(y_true, y_pred, name)
        metrics["mean_er"] = mean_er
        metrics["std_er"] = std_er
        figures["gaussian"] = f

        return metrics, figures, y_pred.detach().cpu().numpy()

    def _mse(self, y_true, y_pred):
        return (y_true - y_pred).pow(2).mean().sqrt().item()

    def _mae(self, y_true, y_pred):
        return (y_true - y_pred).abs().mean().item()

    @profile
    def _energy_resolution_gaussian_fit(self, y_true, y_pred, name):
        normed_predictions = (y_true - y_pred) / y_true
        mean_er = normed_predictions.mean().item()
        std_er = normed_predictions.std().item()
        if LOG_HIST == True:
            f = plt.figure(figsize=(6, 6))
            plt.title("Histogram (E - E_pred) / E, {}".format(name))
            plt.hist(normed_predictions.cpu().detach().numpy(), bins=30, density=True, histtype='step')
            x = [i * 0.01 for i in np.arange(-100, 100)]
            y = norm.pdf(x, mean_er, std_er)
            plt.plot(x, y)
        else:
            f = None
        return mean_er, std_er, f

    @profile
    def log_er_plot(self, metrics, type):
        er = [metrics[i]["std_er"] for i in range(11)]

        f = plt.figure(figsize=(6, 6))
        plt.title("ER plot, Type {}".format(type))
        plt.plot(np.arange(11), er)
        plt.ylabel(r"\sigma / E")
        plt.xlabel(r"E (MeV)")
        return f


class CometLogger(Logger):
    """
    Comet ml logger
    """

    def __init__(self, experiment: CometExperiment):
        self._experiment = experiment
        super(CometLogger, self).__init__()

    @profile
    def log_metrics(self, net, loader, name, device):
        metrics, figures, predictions = super(CometLogger, self).log_metrics(net, loader, name, device)
        if LOG_HIST == True:
            self._experiment.log_figure("Histogram (E - E_pred) / E, {}".format(name), figures["gaussian"], overwrite=True)
        self._experiment.log_metric("MSE, {}".format(name), metrics["mse"])
        self._experiment.log_metric("MAE, {}".format(name), metrics["mae"])
        self._experiment.log_metric("Gaussian mean for (E - E_pred) / E, {}".format(name), metrics["mean_er"])
        self._experiment.log_metric("Gaussian std for (E - E_pred) / E, {}".format(name), metrics["std_er"])
        plt.close(figures["gaussian"])

        return metrics, figures, predictions

    @profile
    def log_er_plot(self, metrics, type):
        f = super(CometLogger, self).log_er_plot(metrics, type)
        self._experiment.log_figure("Energy resolution", f, overwrite=True)
        plt.close(f)
