import os
import yaml
from typing import List


def load_configuration(config_path: str) -> dict:
    """
    Load the configuration file and return the configuration parameters as a dictionary.

    :param config_path: str, path to the YAML configuration file.
    :return: dict, containing configuration parameters.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict, required_keys: List[str]) -> None:
    """
    Validate the configuration parameters.

    :param required_keys:
    :param config: dict, containing configuration parameters.
    :return: None
    """

    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"Configuration file is missing the required parameter: {key}"
            )


def validate_data_format(array_name: str) -> None:
    """
    Validate the data and labels file formats.

    :param array_name: str, name of the data file.
    :return: None
    """
    valid_formats = [".npy", ".pt"]

    if not any(array_name.endswith(fmt) for fmt in valid_formats):
        raise ValueError("Array must be in the form of a numpy array or torch tensor")


def check_required_file(dir_address: str, file_name: str) -> None:
    """
    Check if the required files exist.

    :param dir_address: str, path to the source directory.
    :param file_name: str, name of the file.
    :return: None
    """
    if not os.path.exists(os.path.join(dir_address, file_name)):
        raise FileNotFoundError(f"{file_name} not found in {dir_address}")
