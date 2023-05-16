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

    n_swaps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    # n_swaps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]

    # Load the experiment
    # load_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_4_cache_dir"
    load_save_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_4_afterfix_cache_dir"

    # append the seed and the device to the load path with .pkl ending
    load_path = os.path.join(load_save_path, f"seed_{seed}_device_{device}.pkl")
    with open(load_path, 'rb') as f:
        e_temp = pickle.load(f)


    for ns in n_swaps:
        temp_node = e_temp.get_n_swaps_nodes(n_swaps=ns)[0]

        e_temp.entropy_measure_node(node=temp_node, meas_params=meas_params_ee,
                                    surrogate=False, fold=None)
        e_temp.entropy_measure_node(node=temp_node, meas_params=meas_params_surrogate,
                                    surrogate=True, fold=None)

    # Save the experiment
    # save_path = "/home/fodl/nimrodd_new/dl_unstr_qe_code_base_v1/experiments_reproduction/cache_dirs/figure_4_cache_dir"

    # append the seed and the device to the save path with .pkl ending
    save_path = os.path.join(load_save_path, f"seed_{seed}_device_{device}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(e_temp, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with given seed and device.")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment.")
    parser.add_argument("--device", type=int, help="Device to run the experiment on (e.g., 0).")

    args = parser.parse_args()
    main(args.seed, args.device)