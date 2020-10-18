from comet_ml import Experiment
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import click
from utils.utils import perform_epoch
from logger.logger import CometLogger
from utils.dataloader import JunoLoader
from models.regression_net import RegressionNet
from utils import loss_functions
import numpy as np
import os


def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


@click.command()
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--nonlinearity', type=str, default='ReLU')
@click.option('--loss_function', type=str, default='mse')
@click.option('--hidden_dim', type=int, default=32)
@click.option('--num_hidden', type=int, default=4)
@click.option('--lr', type=float, default=1e-3)
@click.option('--scheduler_type', type=str, default="ReduceLROnPlateau")
@click.option('--datadir', type=str, default='./')
@click.option('--batch_size', type=int, default=512)
@click.option('--epochs', type=int, default=1000)
def train(
        project_name, work_space, datadir="./",
        batch_size=512, lr=1e-3, epochs=1000, nonlinearity="ReLU",
        hidden_dim=20, num_hidden=4, scheduler_type="ReduceLROnPlateau",
        loss_function="mse"
):
    experiment = Experiment(project_name=project_name, workspace=work_space)

    logger = CometLogger(experiment)

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(get_freer_gpu()))
    else:
        device = torch.device('cpu')
    print("Using device = {}".format(device))

    train_filename = os.path.join(datadir, 'ProcessedTrain0.csv')

    juno_loader = JunoLoader().fit(train_filename)
    X, y = juno_loader.transform(train_filename)
    X = torch.tensor(X).float()  # .to(device)
    y = torch.tensor(y).float()  # .to(device)

    dataset = TensorDataset(X, y)
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset,
        (int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)),
    )

    dataloader_kwargs = {}  # {'num_workers': 0, 'pin_memory': True} if USE_CUDA else {}
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    )

    net = RegressionNet(
        input_shape=X.shape[1], output_size=1,
        hidden_dim=hidden_dim, num_hidden=num_hidden,
        nonlinearity=nonlinearity
    ).to(device)

    loss_function = getattr(loss_functions, loss_function)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    best_weights = None
    best_loss = 1e3
    key = experiment.get_key()
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        experiment.set_epoch(epoch)
        mean_loss_train = perform_epoch(net, train_loader, loss_function, device=device, optimizer=optimizer)
        mean_loss_val = perform_epoch(net, val_loader, loss_function, device=device)
        logger.log_metrics(net, train_loader, "Train", device)
        logger.log_metrics(net, val_loader, "Validation", device)
        experiment.log("validation_loss", mean_loss_val)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
            else:
                scheduler.step(mean_loss_val)
        # save weights
        if mean_loss_val < best_loss:
            best_loss = mean_loss_val
            best_weights = net.state_dict()
            torch.save(best_weights, './juno_net_weights_{}.pt'.format(key))
            experiment.log_model('juno_net_weights_{}.pt'.format(key), './juno_net_weights_{}.pt'.format(key))

        if epoch % 10:
            test_metrics = []
            for i in range(11):
                test_filename = os.path.join(datadir, './ProcessedTest{}.csv'.format(i))
                X_test, y_test = juno_loader.transform(test_filename)
                X_test = torch.tensor(X_test).float()  # .to(device)
                y_test = torch.tensor(y_test).float()  # .to(device)
                dataset_test = TensorDataset(X_test, y_test)
                test_loader = torch.utils.data.DataLoader(
                    dataset_test, batch_size=batch_size, shuffle=False, **dataloader_kwargs
                )
                metrics, figures = logger.log_metrics(net, test_loader, "Test, {} MeV".format(i), device)
                test_metrics.append(metrics)
            logger.log_er_plot(test_metrics)
            # collect all metrics on test


if __name__ == "__main__":
    train()
