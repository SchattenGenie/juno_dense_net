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
        "lr": {"min": 1e-4, "max": 1e-2, "type": "double"},
        "nonlinearity": {"type": "categorical", "values": ["ReLU", "Tanh"]},
        "hidden_dim": {"min": 16, "max": 256, "type": "integer", "scalingType": "uniform"},
        "num_hidden": {"min": 2, "max": 10, "type": "integer", "scalingType": "uniform"},
        "batch_size": {"min": 128, "max": 1024, "type": "integer", "scalingType": "uniform"},
        "scheduler_type": {"type": "categorical", "values": ["ReduceLROnPlateau", "CosineAnnealingLR", "None"]},
        "loss_function": {"type": "categorical", "values": ["mse", "mae", "energy_resolution_mse"]},
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
--batch_size {batch_size} --epochs {epochs} 
"""

command_cluster = "sbatch -c {0} -t {1} --gpus={2} --job-name={3} run_command.sh"


@click.command()
@click.option('--slurm', type=bool, default=False)
@click.option('--algorithm', type=str, default='random')  # random bayes
@click.option('--slurm_username', type=str, default='vbelavin')
@click.option('--datadir', type=str, default='./')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--max_epochs', type=int, default=5000)
@click.option('--max_processes_in_parallel', type=int, default=3)
def run_optimization(
        project_name, work_space,
        slurm=False, datadir="./", slurm_username="vbelavin",
        algorithm="bayes", max_processes_in_parallel=3,
        max_epochs=5000
):
    optimizer_config["algorithm"] = algorithm
    optimizer = Optimizer(optimizer_config, project_name=project_name)

    processes = []
    for parameters in optimizer.get_parameters():
        command_to_run = base_command.format(
            epochs=max_epochs,
            project_name=project_name,
            work_space=work_space,
            datadir=datadir,
            **parameters["parameters"]
        )
        print(command_to_run)

        # running on slurm
        if slurm:
            pr_count = subprocess.Popen("squeue | grep {} | wc -l".format(slurm_username), shell=True, stdout=subprocess.PIPE)
            out, err = pr_count.communicate()
            while int(out) > max_processes_in_parallel:
                print("Waiting... ")
                time.sleep(60)
                pr_count = subprocess.Popen("squeue | grep {} | wc -l".format(slurm_username), shell=True, stdout=subprocess.PIPE)
                out, err = pr_count.communicate()
            with open("run_command.sh", "w") as file:
                file.write(base_slurm_command.format(command_to_run))
                process = subprocess.Popen(
                    command_cluster.format(1, 60 * 5, 1, "juno_dense_net_opt"),  # 1 cpu, 5 hours, 1 gpu
                    shell=True,
                    close_fds=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                processes.append(process)
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


if __name__ == "__main__":
    run_optimization()
