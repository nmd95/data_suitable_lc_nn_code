import os
import argparse
import pickle

from experiments_reproduction.experiments_module import Experiment

def main(seed, device):
    # Config parameters
    data_dir_path = os.path.join(os.getcwd(), "experiments_reproduction/paper_datasets/isolet")
    dim = 617
    data_dims = [617]
    num_folds = 3
    num_classes = 26
    device = f"cuda:{str(device)}"

    cut_options_cuts = {"cut_method": "metis",
                        "niter_metis": 50000,
                        "ncuts_metis": 10,
                        "metis_recursive": True, "seed_metis": seed, "num_parts": 2}

    rearr_params_cuts = {"igtd": False, "cut_alg": "metis",
                         "cut_options": cut_options_cuts,
                         }

    rearr_params_igtd = {"igtd": True, "no_imp_counter": 0, "no_imp_count_threshold": 20,
                         "no_imp_val_threshold": 0.0, "device": device, "t": 1000,
                         }

    cnn_epochs = 300

    cnn_model_params = {"num_classes": num_classes, "channels_hidden_dim": 32, "num_cnn_layers": 8, "kernel_size": 3,
                        "stride": 3, "use_batchnorm": True, "pool_size": 3, "act_func": "ReLU", "dropout": 0.0}
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

    s4_epochs = 300

    s4_model_params = {
        "d_input": 1,
        "d_output": num_classes,
        "d_model": 64,
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

    lat_epochs = 300

    lat_model_params = {
        "input_dim": 1,
        "max_seq_len": dim,
        "dim": 128,
        "depth": 8,
        "num_classes": num_classes,
        "local_attn_window_size": 25,
        "dim_head": 32,
        "heads": 4,
        "ff_mult": 4,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0
    }
    lat_optimizer_params = {
        "lr": 0.00005,
        "weight_decay": 0.0
    }

    baseline_params_lat = {
        "run_name": "lat_baseline",
        "model_name": "LocalTransformer",
        "model_parameters": lat_model_params,
        "criterion_name": "CrossEntropyLoss",
        "optimizer_name": "Adam",
        "optimizer_parameters": lat_optimizer_params,
        "batch_sizes": [32, 32, 32],
        "device": device,
        "num_classes": num_classes,
        "epochs": lat_epochs,
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
        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_lat)


        if ns == -1:
            rearr_cuts_temp_node = e_temp.rearrange_node(node=temp_node, rearrange_params=rearr_params_cuts)
            rearr_igtd_temp_node = e_temp.rearrange_node(node=temp_node, rearrange_params=rearr_params_igtd)

            e_temp.run_baseline_node(node=rearr_cuts_temp_node, baseline_params=baseline_params_cnn)
            e_temp.run_baseline_node(node=rearr_cuts_temp_node, baseline_params=baseline_params_s4)
            e_temp.run_baseline_node(node=rearr_cuts_temp_node, baseline_params=baseline_params_lat)


            e_temp.run_baseline_node(node=rearr_igtd_temp_node, baseline_params=baseline_params_cnn)
            e_temp.run_baseline_node(node=rearr_igtd_temp_node, baseline_params=baseline_params_s4)
            e_temp.run_baseline_node(node=rearr_igtd_temp_node, baseline_params=baseline_params_lat)
            # [Other baseline runs]

    # Save the experiment
    save_path = "experiments_reproduction/cache_dirs/table_2_isolet_cache_dir"
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
