#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pickle

from experiments_reproduction.experiments_module import Experiment


def main(seed, device):
    # Config parameters
    data_dir_path = os.path.join(os.getcwd(), "experiments_reproduction/paper_datasets/cifar10")
    dim = 1024
    data_dims = [32, 32]
    num_folds = None
    num_classes = 10
    device = f"cuda:{str(device)}"

    cut_options_cuts = {"cut_method": "metis",
                        "niter_metis": 50000,
                        "ncuts_metis": 10,
                        "metis_recursive": True, "seed_metis": seed, "num_parts": 4}

    rearr_params_cuts = {"igtd": False, "cut_alg": "metis",
                         "cut_options": cut_options_cuts,
                         }

    rearr_params_igtd = {"igtd": True, "no_imp_counter": 0, "no_imp_count_threshold": 20,
                         "no_imp_val_threshold": 0.0, "device": device, "t": 1000,
                         }

    cnn_epochs = 500

    cnn_model_params = {"num_classes": num_classes, "channels_hidden_dim": 32, "num_cnn_layers": 8,
                        "kernel_size": (3, 3),
                        "stride": (3, 3), "use_batchnorm": True, "pool_size": (3, 3), "act_func": "ReLU",
                        "dropout": 0.0}
    cnn_optimizer_params = {
        "lr": 0.001,
        "weight_decay": 0.0001
    }

    scheduler_params = {"milestones": [300], "gamma": 0.1}

    baseline_params_cnn = {"run_name": "cnn2d_baseline", "model_name": "TabularRes2DCNN",
                           "model_parameters": cnn_model_params,
                           "criterion_name": "CrossEntropyLoss", "optimizer_name": "Adam",
                           "optimizer_parameters": cnn_optimizer_params, "batch_sizes": [128, 128, 128],
                           "device": device, "num_classes": num_classes,
                           "epochs": cnn_epochs, "weights_save_path": None,
                           "multiple_gpu": False, "scheduler_name": "MultiStepLR",
                           "scheduler_params": scheduler_params}

    # Run experiment
    n_swaps = [-1, 0]
    e_temp = Experiment(seed=seed, data_dims=data_dims, dim=dim,
                        num_folds=num_folds, data_dir_path=data_dir_path)
    for ns in n_swaps:
        temp_node = e_temp.add_n_swaps_structured_permutation(n_swaps=ns)

        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_cnn)

        if ns == -1:
            rearr_cuts_temp_node = e_temp.rearrange_node(node=temp_node, rearrange_params=rearr_params_cuts)
            rearr_igtd_temp_node = e_temp.rearrange_node(node=temp_node, rearrange_params=rearr_params_igtd)

            e_temp.run_baseline_node(node=rearr_cuts_temp_node, baseline_params=baseline_params_cnn)
            e_temp.run_baseline_node(node=rearr_igtd_temp_node, baseline_params=baseline_params_cnn)

    # Save the experiment
    save_path = "experiments_reproduction/cache_dirs/table_3_cache_dir"
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
