import os
from experiments_reproduction.experiments_module import Experiment
import pickle
import argparse


def main(seed, device):
    data_dir_path = os.path.join(os.getcwd(), "experiments_reproduction/paper_datasets/binary_speech_commands_4096")
    dim = 4096
    data_dims = [4096]
    num_folds = None
    num_classes = 2
    device = f"cuda:{str(device)}"

    svd_options = {
        "vidal_noise": 0.0001,
    }
    meas_params_ee = {
        "run_name": "ave_canonical_at_levels",
        "levels": [1, 2, 3, 4, 5, 6],
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
        "levels": [1, 2, 3, 4, 5, 6],
        "sample_size": 5610,
        "device": device,

    }

    cnn_epochs = 300

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
        "model_name": "M5",
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

    s4_epochs = 300

    s4_model_params = {
        "d_input": 1,
        "d_output": num_classes,
        "d_model": 128,
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
        "local_attn_window_size": 10,
        "dim_head": 64,
        "heads": 2,
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


    n_swaps = [2750, 3000, 3250, 3500, 3750, 4000]
    # n_swaps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]

    # Load the experiment
    # load_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_4_cache_dir"
    load_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_4_afterfix_cache_dir"

    # append the seed and the device to the load path with .pkl ending
    load_path = os.path.join(load_path, f"seed_{seed}_device_{device}.pkl")
    with open(load_path, 'rb') as f:
        e_temp = pickle.load(f)


    for ns in n_swaps:
        temp_node = e_temp.add_n_swaps_structured_permutation(n_swaps=ns)

        e_temp.entropy_measure_node(node=temp_node, meas_params=meas_params_ee,
                                    surrogate=False, fold=None)
        e_temp.entropy_measure_node(node=temp_node, meas_params=meas_params_surrogate,
                                    surrogate=True, fold=None)

        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_cnn)
        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_s4)
        e_temp.run_baseline_node(node=temp_node, baseline_params=baseline_params_lat)

    # Save the experiment
    save_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_4_more_swaps_cache_dir"

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