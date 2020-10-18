import os
import json
import numpy as np
from comet_ml import Experiment as CometExperiment
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


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
    def log_metrics(self, net, loader, name: str, device: str):
        y_true = []
        y_pred = []
        with torch.no_grad():
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

        return metrics, figures

    def _mse(self, y_true, y_pred):
        return (y_true - y_pred).pow(2).mean().sqrt().item()

    def _mae(self, y_true, y_pred):
        return (y_true - y_pred).abs().mean().item()

    def _energy_resolution_gaussian_fit(self, y_true, y_pred, name):
        normed_predictions = (y_true - y_pred) / y_true
        mean_er = normed_predictions.mean().item()
        std_er = normed_predictions.std().item()
        f = plt.figure(figsize=(8, 6))
        plt.title("Histogram (E - E_pred), {}".format(name))
        plt.hist(normed_predictions.cpu().detach().numpy(), bins=100, density=True)
        x = [i * 0.01 for i in np.arange(-100, 100)]
        y = norm.pdf(x, mean_er, std_er)
        plt.plot(x, y)
        return mean_er, std_er, f

    def log_er_plot(self, metrics):
        er = [metrics[i]["std_er"] for i in range(11)]

        f = plt.figure(figsize=(8, 6))
        plt.title("ER plot")
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

    def log_metrics(self, net, loader, name, device):
        metrics, figures = super(CometLogger, self).log_metrics(net, loader, name, device)
        self._experiment.log_figure("Histogram (E - E_pred) / E, {}".format(name), figures["gaussian"])
        self._experiment.log_metric("MSE, {}".format(name), metrics["mse"])
        self._experiment.log_metric("MAE, {}".format(name), metrics["mae"])
        self._experiment.log_metric("Gaussian mean for (E - E_pred) / E, {}".format(name), metrics["mean_er"])
        self._experiment.log_metric("Gaussian std for (E - E_pred) / E, {}".format(name), metrics["std_er"])
        plt.close(figures["gaussian"])

        return metrics, figures

    def log_er_plot(self, metrics):
        f = super(CometLogger, self).log_er_plot(metrics)
        self._experiment.log_figure("Energy resolution", f)
        plt.close(f)
