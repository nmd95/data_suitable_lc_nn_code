import argparse
import os
import pickle as pkl

from configs.config_utils import (
    check_required_file,
    load_configuration,
    validate_config,
    validate_data_format,
)
from rearrangement_algorithms.igtd import IGTD
from rearrangement_algorithms.rearrangement_alg import rearrange_from_data

from data_utils import load_data


def main(config_path):
    # Load and validate the configuration file
    config = load_configuration(config_path)
    validate_config(config, ["restructure_method"])

    # Set required keys based on the restructure method
    if config["restructure_method"] == "feature_rearrangement_algorithm":
        required_keys = ["feature_rearrangement_algorithm_parameters", "data_dir_name", "data_name"]
    else:
        required_keys = ["igtd_parameters", "data_dir_name", "data_name"]

    # Validate the configuration parameters
    validate_config(config, required_keys)

    # Construct the data source directory path
    data_dir_address = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../user_datasets", config["data_dir_name"]
    )

    # Check if required files exist
    check_required_file(data_dir_address, config["data_name"])

    # Validate the data and labels file formats
    validate_data_format(config["data_name"])

    # Load the data
    data = load_data(data_dir_address, config["data_name"])


    # Perform rearrangement based on the chosen restructure method
    if config["restructure_method"] == "feature_rearrangement_algorithm":
        rearrangement = perform_feature_rearrangement_algorithm_rearrangement(config, data)
        save_address = os.path.join(data_dir_address, "rearrangement_feature_rearrangement_algorithm.pkl")
    else:
        rearrangement = perform_igtd_rearrangement(config, data)
        save_address = os.path.join(data_dir_address, "rearrangement_igtd.pkl")

    # Save the rearrangement to the data source directory

    save_rearrangement(save_address, rearrangement)


def perform_feature_rearrangement_algorithm_rearrangement(config, data):
    restruct_params = config["feature_rearrangement_algorithm_parameters"]
    required_keys = [
        "cut_method",
        "niter_metis",
        "ncuts_metis",
        "metis_recursive",
        "seed_metis",
        "num_parts",
    ]
    validate_config(restruct_params, required_keys)
    return rearrange_from_data(data=data, cut_options=restruct_params)


def perform_igtd_rearrangement(config, data):
    restruct_params = config["igtd_parameters"]
    required_keys = [
        "no_imp_counter",
        "no_imp_count_threshold",
        "no_imp_val_threshold",
        "device",
        "t",
    ]
    validate_config(restruct_params, required_keys)

    t = restruct_params["t"]
    restruct_params.pop("t")  # Remove the 't' parameter from the dictionary
    restruct_params["data_array"] = data  # Add train_data_flat to the dictionary

    igtd = IGTD(**restruct_params)
    igtd.run(t=t)
    return igtd.get_permutation()


def save_rearrangement(data_dir_address, rearrangement):
    with open(data_dir_address, "wb") as f:
        pkl.dump(rearrangement, f)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Call the main function with the specified configuration file
    main(args.config_path)
