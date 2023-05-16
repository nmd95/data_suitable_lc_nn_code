import argparse
import os
import pickle as pkl
from configs.config_utils import (
    check_required_file,
    load_configuration,
    validate_config,
    validate_data_format,
)
from common.data_utils import (
    get_stable_corr_mat,
    load_data,
    process_labels,
    rearrange_partitions,
)
from entropy.entropy_module import EntropyCalc
import torch


def main(config_path: str):
    """
    Main function that calculates per-level entropies for a given dataset.

    param:
    config_path: str, path to the YAML configuration file.
    """
    # Load the configuration file
    config = load_configuration(config_path)

    validate_config(config, ["entropy"])

    if config["entropy"] == "surrogate":
        required_keys = ["data_dir_name", "data_name", "levels"]
    else:
        required_keys = [
            "data_dir_name",
            "data_name",
            "labels_name",
            "svd_options",
            "theta",
            "levels",
            "entropy",
            "device",
            "fb_size",
            "batch_size",
            "n_batches",
        ]
    # Validate the configuration parameters
    validate_config(config, required_keys)

    # Construct the data source directory path
    data_dir_address = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../user_datasets",
        config["data_dir_name"],
    )

    # Check if required files exist
    check_required_file(data_dir_address, config["data_name"])
    check_required_file(data_dir_address, config["labels_name"])

    # Validate the data and labels file formats
    validate_data_format(config["data_name"])
    validate_data_format(config["labels_name"])

    # Load the data and labels
    data = load_data(data_dir_address, config["data_name"])
    labels = load_data(data_dir_address, config["labels_name"])

    data_mods = len(data.shape[1:])
    data_flat = rearrange_partitions(data)

    labels = process_labels(labels)

    ec = EntropyCalc(
        device=config["device"],
        theta=config["theta"],
        svd_options=config["svd_options"],
    )

    if config["entropy"] == "entanglement_entropy_ee":
        per_level_entropies = ec.canonical_at_levels(
            levels=config["levels"],
            ee=True,
            data=data_flat,
            labels=labels,
            fb_size=config["fb_size"],
            batch_size=config["batch_size"],
            n_batches=config["n_batches"],
            data_mods=data_mods,
        )
    elif config["entropy"] == "entanglement_entropy_ge":
        per_level_entropies = ec.canonical_at_levels(
            levels=config["levels"],
            ee=False,
            data=data_flat,
            labels=labels,
            fb_size=config["fb_size"],
            batch_size=config["batch_size"],
            n_batches=config["n_batches"],
            data_mods=data_mods,
        )
    elif config["entropy"] == "surrogate":
        corr_mat = torch.from_numpy(get_stable_corr_mat(data_flat))
        per_level_entropies = ec.surrogate_canonical_at_levels(
            levels=config["levels"], corr_mat=corr_mat, data_mods=data_mods
        )
    else:
        raise ValueError("Invalid entropy type")

    # Save the rearrangement to the data source directory

    if config["entropy"] == "entanglement_entropy_ee":
        save_name = "per_level_entanglement_entropies.pkl"
    elif config["entropy"] == "entanglement_entropy_ge":
        save_name = "per_level_geometric_entropies.pkl"
    else:
        save_name = "per_level_surrogate_entropies.pkl"

    with open(os.path.join(data_dir_address, save_name), "wb") as f:
        pkl.dump(per_level_entropies, f)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to YAML configuration file")
    args = parser.parse_args()
    # Call the main function with the specified configuration file
    main(args.config_path)
