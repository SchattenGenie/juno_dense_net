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
from utils.utils import perform_epoch, CustomDataLoader
from sklearn.model_selection import train_test_split
from logger.logger import CometLogger, CometLoggerVertex
from utils.dataloader import JunoLoader
from models.regression_net import RegressionNet
from collections import defaultdict
from utils import loss_functions
import pickle
import numpy as np
import os
ENERGY = "energy"
VERTEX = "vertex"
dataset_sizes = {
    "1M": "",
    "5M": "_5M",
}


def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    try:
        return np.argmax(memory_available)
    except ValueError:
        return "0"


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


def logging_test_data_all_types(logger, net, test_data, key, target_variable, device, epoch):
    from collections import defaultdict
    energies = [
            '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8',
            '0.9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
    datatable_predictions = defaultdict(list)
    for type in ["0", "3", "20", "23"]:
        test_metrics = []
        for energy in energies:
            X_test, y_test = test_data[(type, energy)]
            metrics, figures, predictions = logger.log_metrics(
                net,
                (X_test, y_test),
                "Test, Type {}, {} MeV".format(type, energy),
                device,
                step=epoch
            )
            if target_variable == ENERGY: shape = (-1, )
            elif target_variable == VERTEX: shape = (3, -1)
            datatable_predictions[(type, energy)] = np.vstack([y_test.detach().cpu().numpy().reshape(*shape),
                                                               predictions.reshape(*shape)]).T
            test_metrics.append(metrics)
        logger.log_er_plot(energies, test_metrics, type, step=epoch)
    with open("datatable_predictions_{}.pkl".format(key), 'wb') as f:
        pickle.dump(datatable_predictions, f)
    logger._experiment.log_asset("datatable_predictions_{}.pkl".format(key), overwrite=True, copy_to_tmp=False, step=epoch)


@click.command()
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--nonlinearity', type=str, default='ReLU')
@click.option('--loss_function', type=str, default='mse')
@click.option('--hidden_dim', type=int, default=32)
@click.option('--num_hidden', type=int, default=7)
@click.option('--lr', type=float, default=2e-3)
@click.option('--dropout', type=float, default=0.0)
@click.option('--scheduler_type', type=str, default="CosineAnnealingLR")
@click.option('--use_swa', type=bool, default=False)
@click.option('--use_layer_norm', type=bool, default=False)
@click.option('--optimizer_cls', type=str, default="Adam")
@click.option('--init_type', type=str, default="normal")
@click.option('--target_variable', type=str, default="energy")  # energy, vertex
@click.option('--train_type', type=str, default="0")  # 0 20 3 23
@click.option('--datadir', type=str, default='./')
@click.option('--batch_size', type=int, default=600)
@click.option('--epochs', type=int, default=3000)
@click.option('--dataset_size', type=str, default="1M")
@click.option('--coeffs', type=str, default="1.,0.,0.,0.")
def train(
        project_name, work_space, datadir="./", train_type="0", dataset_size="1M",
        batch_size=512, lr=1e-3, epochs=1000, nonlinearity="ReLU",
        hidden_dim=20, num_hidden=4, scheduler_type="ReduceLROnPlateau",
        loss_function="mse", use_swa=False, optimizer_cls="Adam", dropout=0.,
        use_layer_norm=False, init_type="normal", target_variable=ENERGY, coeffs="1.,0.,0.,0."
):
    # comet logger instance preparation
    experiment = Experiment(
        project_name=project_name, workspace=work_space,
        auto_metric_logging=False, auto_log_co2=False, auto_output_logging="simple"
    )
    if target_variable == ENERGY:
        logger = CometLogger(experiment)
    elif target_variable == VERTEX:
        logger = CometLoggerVertex(experiment)
    else:
        raise ValueError()

    # initialization of cuda device
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(get_freer_gpu()))
    else:
        device = torch.device('cpu')
    print("Using device = {}".format(device))

    coeffs = [float(x.strip()) for x in coeffs.split(',')]

    # data preparation
    # all data is stored on gpu, because it weights not so much
    train_filename = os.path.join(datadir, 'ProcessedTrainReduced{}{}.csv'.format(train_type, dataset_sizes[dataset_size]))
    juno_loader = JunoLoader(target_variable=target_variable)
    X, y, energy = juno_loader.transform(train_filename)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).float().to(device)
    energy = torch.tensor(energy).float().to(device)

    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.1, random_state=42, shuffle=True)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    energy_train, energy_val = energy[train_idx], energy[val_idx]

    train_loader = CustomDataLoader(X_train, y_train, energy_train, batch_size=batch_size)
    val_loader = CustomDataLoader(X_val, y_val, energy_val, batch_size=batch_size)

    # dataset_train = TensorDataset(X_train, y_train)
    # dataset_val = TensorDataset(X_val, y_val)
    # dataloader_kwargs = {'num_workers': 0, 'pin_memory': True} if USE_CUDA else {}
    # train_loader = torch.utils.data.DataLoader(
    #    dataset_train, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    # )
    # val_loader = torch.utils.data.DataLoader(
    #    dataset_val, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    # )

    # loading test data
    test_data = defaultdict(list)
    for type in ["0", "3", "20", "23"]:
        for energy in [
            '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8',
            '0.9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'
        ]:
            test_filename = os.path.join(datadir, './ProcessedTestReduced{}/{}MeV.csv'.format(type, energy))
            X_test, y_test, _ = juno_loader.transform(test_filename)
            X_test = torch.tensor(X_test).float()
            y_test = torch.tensor(y_test).float()
            test_data[(type, energy)] = (X_test, y_test)

    # setting up network
    if target_variable == ENERGY:
        output_size = 1
    elif target_variable == VERTEX:
        output_size = 3
    net = RegressionNet(
        input_shape=X.shape[1], output_size=output_size,
        hidden_dim=hidden_dim, num_hidden=num_hidden,
        nonlinearity=nonlinearity, layer_norm=use_layer_norm,
        init_type=init_type, dropout=dropout
    ).to(device)

    # setting up various optimizations techniques
    loss_function = loss_functions.CombinatorialLoss(coeffs)
    if optimizer_cls == "SGD":
        optimizer = getattr(torch.optim, optimizer_cls)(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = getattr(torch.optim, optimizer_cls)(net.parameters(), lr=lr)

    swa_net = None
    swa_scheduler = None
    swa_start_epoch = int(0.75 * epochs)
    if use_swa:
        swa_net = AveragedModel(net)
        swa_scheduler = SWALR(optimizer, swa_lr=lr)

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    best_weights = None
    best_loss = 1e3
    not_yet_logged = True
    last_logged = 0
    throttling_pace = 8
    key = experiment.get_key()
    experiment.set_step(0)
    for epoch in tqdm(range(epochs)):
        print(last_logged, not_yet_logged)
        print("Epoch {}".format(epoch))
        experiment.set_step(1 + epoch)
        _ = perform_epoch(net, train_loader, loss_function, device=device, optimizer=optimizer, epoch=epoch)
        if use_swa and epoch > swa_start_epoch:
            mean_loss_val = perform_epoch(swa_net, val_loader, loss_function, device=device)
            logger.log_metrics(swa_net, train_loader, "Train", device, step=epoch)
            logger.log_metrics(swa_net, val_loader, "Validation", device, step=epoch)
        else:
            mean_loss_val = perform_epoch(net, val_loader, loss_function, device=device)
            logger.log_metrics(net, train_loader, "Train", device, step=epoch)
            logger.log_metrics(net, val_loader, "Validation", device, step=epoch)

        experiment.log_metric("validation_loss", mean_loss_val, step=epoch)
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
            net.eval()
            not_yet_logged = True
            best_loss = mean_loss_val
            if use_swa and epoch > swa_start_epoch:
                best_weights = swa_net.state_dict()
            else:
                best_weights = net.state_dict()

            # to not be banned by comet.ml
            if last_logged < throttling_pace:
                last_logged += 1
                continue

            torch.save(best_weights, './juno_net_weights_{}.pt'.format(key))
            experiment.log_model(
                'juno_net_weights_{}.pt'.format(key), './juno_net_weights_{}.pt'.format(key), overwrite=True
            )
            if use_swa and epoch > swa_start_epoch:
                logging_test_data_all_types(logger=logger, net=swa_net, test_data=test_data, key=key, device=device, target_variable=target_variable, epoch=epoch)
            else:
                logging_test_data_all_types(logger=logger, net=net, test_data=test_data, key=key, device=device, target_variable=target_variable, epoch=epoch)
            last_logged = 0
            not_yet_logged = False
        last_logged += 1

        # logging every 10 epochs or so
        if not_yet_logged and last_logged >= throttling_pace:
            net.eval()
            torch.save(best_weights, './juno_net_weights_{}.pt'.format(key))
            experiment.log_model(
                'juno_net_weights_{}.pt'.format(key), './juno_net_weights_{}.pt'.format(key), overwrite=True
            )
            if use_swa and epoch > swa_start_epoch:
                logging_test_data_all_types(logger=logger, net=swa_net, test_data=test_data, key=key, device=device, target_variable=target_variable, epoch=epoch)
            else:
                logging_test_data_all_types(logger=logger, net=net, test_data=test_data, key=key, device=device, target_variable=target_variable, epoch=epoch)
            last_logged = 0
            not_yet_logged = False

if __name__ == "__main__":
    train()
