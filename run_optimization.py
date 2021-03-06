import random
import comet_ml
from typing import List, Tuple
import time
from copy import copy, deepcopy
from comet_ml import Optimizer
import subprocess
import numpy as np
import shlex
import click
import os


optimizer_config = {
    "algorithm": "bayes",
    "spec": {
        "maxCombo": 100,
        "metric": "validation_loss"
    },
    "parameters": {
        "lr": {"min": 2e-3, "max": 2e-3, "type": "double", "scalingType": "loguniform"},
        "dropout": {"min": 0., "max": 0.0, "type": "double", "scalingType": "uniform"},
        "nonlinearity": {"type": "categorical", "values": ["ReLU"]},  # "Tanh"
        "hidden_dim": {"min": 32, "max": 32, "type": "integer", "scalingType": "uniform"},
        "num_hidden": {"min": 7, "max": 7, "type": "integer", "scalingType": "uniform"},
        "batch_size": {"type": "categorical", "values": ["600"]},
        "scheduler_type": {"type": "categorical", "values": ["CosineAnnealingLR"]},  # , "None", "ReduceLROnPlateau"
        "loss_function": {"type": "categorical", "values": ["CombinatorialLoss"]},
        "use_layer_norm": {"type": "categorical", "values": ["False"]},  # "True",
        # "mse", "mae", "energy_resolution_mse", "energy_resolution_sqrt", "energy_resolution_mse_shifted",
        # "energy_resolution_mse_with_mse", "energy_resolution_mae"
        "use_swa": {"type": "categorical", "values": ["False"]},  # "True",
        "optimizer_cls": {"type": "categorical", "values": ["Adam"]},  # "Adagrad", "SGD", "RMSprop"
        "init_type": {"type": "categorical", "values": ["normal"]},  # , "uniform", "orthogonal"
        "seed": {"min": 0, "max": 1000, "type": "integer", "scalingType": "uniform"}
        # "epochs": {"type": "categorical", "values": [500, 1000, 2000, 3000, 4000]}
    },
}


base_slurm_command = """#!/bin/bash
set -x
{0}
"""

base_command = """python train_model.py --project_name {project_name} \
--work_space {work_space} --datadir {datadir} \
--lr {lr} --hidden_dim {hidden_dim} --num_hidden {num_hidden} \
--nonlinearity {nonlinearity} --scheduler_type {scheduler_type} \
--batch_size {batch_size} --epochs {epochs} --use_swa {use_swa} \
--optimizer_cls {optimizer_cls} --use_layer_norm {use_layer_norm} \
--init_type {init_type} --train_type {train_type} --loss_function {loss_function} \
--coeffs {coeffs} --target_variable {target_variable} --dropout {dropout}"""

command_cluster = "sbatch -c {0} -t {1} --gpus={2} --job-name={3} run_command.sh"


@click.command()
@click.option('--slurm', type=bool, default=False)
@click.option('--algorithm', type=str, default='random')  # random bayes
@click.option('--slurm_username', type=str, default='vbelavin')
@click.option('--datadir', type=str, default='./')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--max_epochs', type=int, default=3000)
@click.option('--max_processes_in_parallel', type=int, default=3)
@click.option('--train_type', type=str, default="0")  # 0 20 3 23
@click.option('--train_nets_on_one_gpu', type=int, default=10)  # only for slurm
@click.option('--target_variable', type=str, default="energy")  # energy, vertex
def run_optimization(
        project_name, work_space,
        slurm=False, datadir="./", slurm_username="vbelavin",
        algorithm="bayes", max_processes_in_parallel=3,
        train_nets_on_one_gpu=3, max_epochs=5000, train_type="0",
        target_variable="energy"
):
    optimizer_config["algorithm"] = algorithm
    optimizer = Optimizer(optimizer_config, project_name=project_name)

    processes = []
    commands_to_run = []
    for parameters in optimizer.get_parameters():
        x = np.diff(np.sort(np.random.uniform(size=5)))
        x = x / x.sum()
        command_to_run = base_command.format(
            epochs=max_epochs,
            project_name=project_name,
            work_space=work_space,
            datadir=datadir,
            train_type=train_type,
            target_variable=target_variable,
            coeffs="{},{},{},{}".format(x[0], x[1], x[2], x[3]),
            **parameters["parameters"]
        )
        print(command_to_run)
        commands_to_run.append(command_to_run)

        # running on slurm
        if slurm:
            pr_count = subprocess.Popen("squeue | grep {} | wc -l".format(slurm_username), shell=True, stdout=subprocess.PIPE)
            out, err = pr_count.communicate()
            while int(out) > max_processes_in_parallel:
                print("Waiting... ")
                time.sleep(60)
                pr_count = subprocess.Popen("squeue | grep {} | wc -l".format(slurm_username), shell=True, stdout=subprocess.PIPE)
                out, err = pr_count.communicate()
            if len(commands_to_run) >= train_nets_on_one_gpu:
                with open("run_command.sh", "w") as file:
                    file.write(base_slurm_command.format(" &\n".join(commands_to_run) + " &\nwait"))
                    process = subprocess.Popen(
                        command_cluster.format(4, 60 * 30, 1, "juno_dense_net_opt"),  # 3 cpu, 30 hours, 1 gpu
                        shell=True,
                        close_fds=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    processes.append(process)
                commands_to_run = []

        # running on local machine via Popen
        else:
            while len(processes) > max_processes_in_parallel:
                print("Waiting... ")
                time.sleep(60)
                processes_after_cleaning = []
                for process in processes:
                    poll = process.poll()
                    if poll is not None:
                        processes_after_cleaning.append(process)
                processes = processes_after_cleaning
            command_to_run = shlex.split(command_to_run)
            process = subprocess.Popen(
                command_to_run,
                shell=False,
                close_fds=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            processes.append(process)
            commands_to_run = []


if __name__ == "__main__":
    run_optimization()
