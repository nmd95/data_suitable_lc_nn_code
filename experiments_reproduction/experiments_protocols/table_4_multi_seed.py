import os
import numpy as np
from common.data_utils import get_stable_corr_mat
from experiments_reproduction.experiments_module import Experiment
import pickle
import argparse
import subprocess


def main(seed, device):
    # Config parameters

    data_dir_path = os.path.join(os.getcwd(), "experiments_reproduction/paper_datasets/speech_commands_dim_50000")
    dim = 50000
    data_dims = [50000]
    num_folds = None
    num_classes = 35
    device = f"cuda:{str(device)}"

    cut_options_cuts = {"cut_method": "metis",
                        "niter_metis": 50000,
                        "ncuts_metis": 10,
                        "metis_recursive": True, "seed_metis": None, "num_parts": 2}

    rearr_params_cuts = {"igtd": False, "cut_alg": "metis",
                         "cut_options": cut_options_cuts,
                         "sparse": True
                         }

    cnn_epochs = 200

    cnn_model_params = {
        "n_input": 1,
        "n_output": num_classes,
        "kernel_size": 80,
        "stride": 4,
        "n_channel": 128
    }
    cnn_optimizer_params = {
        "lr": 0.001,
        "weight_decay": 0.0001
    }

    baseline_params_cnn = {
        "run_name": "cnn_baseline",
        "model_name": "TabularResCNN",
        "model_parameters": cnn_model_params,
        "criterion_name": "CrossEntropyLoss",
        "optimizer_name": "Adam",
        "optimizer_parameters": cnn_optimizer_params,
        "batch_sizes": [64, 64, 64],
        "device": device,
        "num_classes": num_classes,
        "epochs": cnn_epochs,
        "weights_save_path": None,
        "multiple_gpu": False
    }

    s4_epochs = 200

    s4_model_params = {
        "d_input": 1,
        "d_output": num_classes,
        "d_model": 32,
        "dropout": 0.0,
        "prenorm": False,
        "lr": 0.001,
        "n_layers": 4
    }
    s4_setup_params = {
        "lr": 0.001,
        "weight_decay": 0.01,
        "epochs": s4_epochs
    }

    baseline_params_s4 = {
        "run_name": "s4_baseline",
        "model_name": "S4Model",
        "model_parameters": s4_model_params,
        "criterion_name": "CrossEntropyLoss",
        "s4_setup_params": s4_setup_params,
        "batch_sizes": [64, 64, 64],
        "device": device,
        "num_classes": num_classes,
        "epochs": s4_epochs,
        "weights_save_path": None,
        "multiple_gpu": False
    }

    # Run experiment

    n_swaps = [-1, 0]

    e_temp = Experiment(seed=seed, data_dims=data_dims, dim=dim,
                        num_folds=num_folds, data_dir_path=data_dir_path)
    for ns in n_swaps:
        temp_node = e_temp.add_n_swaps_structured_permutation(n_swaps=ns)

        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_cnn)
        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_s4)

        if ns == -1:
            rearr_params_cuts['cut_options']['seed_metis'] = seed
            rearr_cuts_temp_node = e_temp.rearrange_node(node=temp_node, rearrange_params=rearr_params_cuts)

            e_temp.run_baseline_node(node=rearr_cuts_temp_node, baseline_params=baseline_params_cnn)
            e_temp.run_baseline_node(node=rearr_cuts_temp_node, baseline_params=baseline_params_s4)

    # Save the experiment
    save_path = "experiments_reproduction/cache_dirs/table_4_cache_dir"
    # append the seed and the device to the save path
    save_path = os.path.join(save_path, f"seed_{seed}_device_{device}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(e_temp, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run experiment with given seed and device.")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment.")
    parser.add_argument("--device", type=int, help="Device to run the experiment on (e.g., 0).")

    args = parser.parse_args()
    main(args.seed, args.device)


