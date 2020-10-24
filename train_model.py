from comet_ml import Experiment
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
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
import pickle
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


def logging_test_data_all_types(datadir, logger, net):
    datatable_predictions = defaultdict(list)
    for type in ["0", "3", "20", "23"]:
        test_metrics = []
        for energy in [
            '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8',
            '0.9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'
        ]:
            test_filename = os.path.join(datadir, './ProcessedTestReduced{}/{}.csv'.format(type, energy))
            X_test, y_test = juno_loader.transform(test_filename)
            X_test = torch.tensor(X_test).float()
            y_test = torch.tensor(y_test).float()
            dataset_test = TensorDataset(X_test, y_test)
            test_loader = torch.utils.data.DataLoader(
                dataset_test, batch_size=1024, shuffle=False, **dataloader_kwargs
            )
            metrics, figures, predictions = logger.log_metrics(
                net, test_loader, "Test, Type {}, {} MeV".format(type, energy),
                device
            )
            datatable_predictions[(type, energy)] = np.vstack([y_test.detach().cpu().numpy(), predictions]).T
            test_metrics.append(metrics)
        logger.log_er_plot(test_metrics)
    if upload_pickle:
        with open("datatable_predictions.pkl", 'wb') as f:
            pickle.dump(datatable_predictions, f)
        logger._experiment.log_asset("datatable_predictions.pkl", overwrite=True, copy_to_tmp=False)


@click.command()
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--nonlinearity', type=str, default='ReLU')
@click.option('--loss_function', type=str, default='mse')
@click.option('--hidden_dim', type=int, default=32)
@click.option('--num_hidden', type=int, default=4)
@click.option('--lr', type=float, default=1e-3)
@click.option('--scheduler_type', type=str, default="ReduceLROnPlateau")
@click.option('--use_swa', type=bool, default=False)
@click.option('--use_layer_norm', type=bool, default=False)
@click.option('--optimizer_cls', type=str, default="Adam")
@click.option('--init_type', type=str, default="normal")
@click.option('--datadir', type=str, default='./')
@click.option('--batch_size', type=int, default=512)
@click.option('--epochs', type=int, default=1000)
def train(
        project_name, work_space, datadir="./",
        batch_size=512, lr=1e-3, epochs=1000, nonlinearity="ReLU",
        hidden_dim=20, num_hidden=4, scheduler_type="ReduceLROnPlateau",
        loss_function="mse", use_swa=False, optimizer_cls="Adam",
        use_layer_norm=False, init_type="normal"
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
        nonlinearity=nonlinearity, layer_norm=use_layer_norm,
        init_type=init_type
    ).to(device)

    loss_function = getattr(loss_functions, loss_function)
    if optimizer_cls == "SGD":
        optimizer = getattr(torch.optim, optimizer_cls)(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = getattr(torch.optim, optimizer_cls)(net.parameters(), lr=lr)

    swa_net = None
    swa_scheduler = None
    swa_start_epoch = int(0.75 * epochs)
    if use_swa:
        swa_net = AveragedModel(net)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

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
        _ = perform_epoch(net, train_loader, loss_function, device=device, optimizer=optimizer)
        if use_swa and epoch > swa_start_epoch:
            mean_loss_val = perform_epoch(swa_net, val_loader, loss_function, device=device)
            logger.log_metrics(swa_net, train_loader, "Train", device)
            logger.log_metrics(swa_net, val_loader, "Validation", device)
        else:
            mean_loss_val = perform_epoch(net, val_loader, loss_function, device=device)
            logger.log_metrics(net, train_loader, "Train", device)
            logger.log_metrics(net, val_loader, "Validation", device)

        experiment.log_metric("validation_loss", mean_loss_val)
        if use_swa and epoch > swa_start_epoch:
            swa_net.update_parameters(net)
            swa_scheduler.step()
            torch.optim.swa_utils.update_bn(train_loader, swa_net)
        else:
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                else:
                    scheduler.step(mean_loss_val)

        # save weights and test on test
        if mean_loss_val < best_loss:
            best_loss = mean_loss_val
            if use_swa and epoch > swa_start_epoch:
                best_weights = swa_net.state_dict()
            else:
                best_weights = net.state_dict()
            torch.save(best_weights, './juno_net_weights_{}.pt'.format(key))
            experiment.log_model(
                'juno_net_weights_{}.pt'.format(key), './juno_net_weights_{}.pt'.format(key), overwrite=True
            )
            if use_swa and epoch > swa_start_epoch:
                logging_test_data_all_types(datadir=datadir, logger=logger, net=swa_net)
            else:
                logging_test_data_all_types(datadir=datadir, logger=logger, net=net)


if __name__ == "__main__":
    train()
