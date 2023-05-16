#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pickle
# print python executing location
from experiments_reproduction.experiments_module import Experiment


def main(seed, device):
    # Config parameters

    data_dir_path = os.path.join(os.getcwd(), "experiments_reproduction/paper_datasets/cifar10_binary")
    dim = 1024
    data_dims = [32, 32]
    num_folds = None
    num_classes = 2
    device = f"cuda:{str(device)}"

    svd_options = {
        "vidal_noise": 0.0001,
    }
    meas_params_ee = {
        "run_name": "ave_canonical_at_levels",
        "levels": [1, 2],
        "ee": True,
        "sample_size": 1000,
        "device": device,
        "fb_size": 500,
        "const": 0,
        "theta": 0.085,
        "svd_options": svd_options,
        "batch_size": 500,
        "n_batches": 2
    }
    meas_params_surrogate = {
        "run_name": "ave_surrogate_canonical_at_levels",
        "levels": [1, 2],
        "sample_size": 1000,
        "device": device,
    }

    cnn_epochs = 150

    cnn_model_params = {"num_classes": num_classes, "channels_hidden_dim": 32, "num_cnn_layers": 8,
                        "kernel_size": (3, 3),
                        "stride": (3, 3), "use_batchnorm": True, "pool_size": (3, 3), "act_func": "ReLU",
                        "dropout": 0.0}
    cnn_optimizer_params = {
        "lr": 0.001,
        "weight_decay": 0.0001
    }

    baseline_params_cnn = {
        "run_name": "cnn_baseline",
        "model_name": "TabularRes2DCNN",
        "model_parameters": cnn_model_params,
        "criterion_name": "CrossEntropyLoss",
        "optimizer_name": "Adam",
        "optimizer_parameters": cnn_optimizer_params,
        "batch_sizes": [128, 128, 128],
        "device": device,
        "num_classes": num_classes,
        "epochs": cnn_epochs,
        "weights_save_path": None,
        "multiple_gpu": False
    }

    # Run experiment
    n_swaps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]

    e_temp = Experiment(seed=seed, data_dims=data_dims, dim=dim,
                        num_folds=num_folds, data_dir_path=data_dir_path)
    for ns in n_swaps:
        temp_node = e_temp.add_n_swaps_structured_permutation(n_swaps=ns)

        e_temp.entropy_measure_node(node=temp_node, meas_params=meas_params_ee,
                                    surrogate=False, fold=None)
        e_temp.entropy_measure_node(node=temp_node, meas_params=meas_params_surrogate,
                                    surrogate=True, fold=None)

        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_cnn)

    # Save the experiment
    save_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_8_cache_dir"
    # append the seed and the device to the save path with .pkl ending
    save_path = os.path.join(save_path, f"seed_{seed}_device_{device}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(e_temp, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with given seed and device.")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment.")
    parser.add_argument("--device", type=int, help="Device to run the experiment on (e.g., 0).")

    args = parser.parse_args()
    main(args.seed, args.device)

