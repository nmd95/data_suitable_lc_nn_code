import itertools
import os
from typing import List, Union

import numpy as np
import torch
import subprocess

def create_index_map(array_shape: tuple) -> np.ndarray:
    """
    Generates index mapping for partitioning an array into halves along each axis recursively.

    Args:
        array_shape (tuple): The shape of the array to be partitioned.

    Returns:
        np.ndarray: The index mapping array.
    """
    num_elements = np.prod(array_shape)
    arr = np.arange(num_elements).reshape(array_shape)
    # unsqueeze to add a batch dimension
    arr = np.expand_dims(arr, axis=0)
    rearranged_arr = rearrange_partitions(arr)
    # squeeze to remove the batch dimension
    rearranged_arr = np.squeeze(rearranged_arr, axis=0)
    return np.argsort(rearranged_arr).reshape(array_shape)


def rearrange_partitions(
    input_array: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Rearrange the partitions of a given input array or tensor by recursively dividing it into sub-arrays or
    sub-tensors by halving the dimension of each axis and concatenating them to form a flattened array or tensor.

    This function supports both numpy arrays and PyTorch tensors.

    Parameters
    ----------
    input_array : Union[np.ndarray, torch.Tensor]
        The input array or tensor to be rearranged. The array or tensor must have at least two dimensions, where the
        first dimension is the batch dimension.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The rearranged array or tensor with the same type as the input.

    Notes
    -----
    This function is being used to treat multidimensional arrays or tensors as flattened arrays or tensors with their
    canonical partitions obtained by dividing the array into 2^p equal, where p is the number of axis of the tensor.
    """

    # Determine if the input is a torch.Tensor or a numpy array
    is_torch = isinstance(input_array, torch.Tensor)
    # Set the module to be used for tensor manipulation based on the input type
    module = torch if is_torch else np

    # Get the shape of the input_array
    shape = input_array.shape
    # If any dimension (except the first one) is less than or equal to 2, place its element contiguously
    if any(s <= 2 for s in shape[1:]):
        return input_array.reshape(shape[0], -1)

    # Initialize an empty list to store half_spaces for each dimension (except the first one)
    # Half space refer to the two halves of a dimension
    half_spaces = []
    for s in shape[1:]:
        # Calculate the half_space for the current dimension
        half_space = [(0, s // 2), (s // 2, s)]
        half_spaces.append(half_space)

    # Generate the cartesian products of the half_spaces
    cartesian_products = list(itertools.product(*half_spaces))

    # Initialize an empty list to store the sub-arrays or sub-tensors
    sub_arrays = []
    for product in cartesian_products:
        # Create the slices for the input_array
        slices = (slice(None),) + tuple(slice(start, end) for start, end in product)
        # Get the sub-array or sub-tensor using the slices
        sub_array = input_array[slices]
        # Recursively apply the rearrange_partitions function on the sub-array or sub-tensor
        sub_arrays.append(rearrange_partitions(sub_array))

    # Concatenate the sub-arrays or sub-tensors along the second dimension (axis=1)
    return module.concatenate(sub_arrays, axis=1)


def permute_elements(
    input_array: Union[np.ndarray, torch.Tensor], permutation: List[int]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Permute elements of an input array based on the given permutation.

    Args:
        input_array (np.ndarray or torch.Tensor): An input batch array with a leading batch dimension and an arbitrary number of axes.
        permutation (List[int]): A list of integers representing the desired permutation.

    Returns:
        np.ndarray or torch.Tensor: The permuted input array.
    """
    temp_array = rearrange_partitions(input_array)
    temp_array = temp_array[:, permutation]
    index_map = create_index_map(input_array.shape[1:])
    if isinstance(input_array, torch.Tensor):
        index_map = torch.tensor(index_map)
    temp_array = temp_array[:, index_map]
    return temp_array.reshape(input_array.shape)


def load_data(data_dir_address: str, data_name: str) -> torch.Tensor:
    """
    Load the data and labels.

    :param data_dir_address: str, path to the data source directory.
    :param data_name: str, name of the data file.
    :return: torch tensors, data.
    """
    data_path = os.path.join(data_dir_address, data_name)

    if data_name.endswith(".npy"):
        data = np.load(data_path)
        data = torch.from_numpy(data)
    else:
        data = torch.load(data_path)

    return data


def process_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Check if labels are binary and map them to -1 and 1.

    :param labels: torch.tensor, input labels.
    :return: torch.tensor, processed labels with values -1 and 1.
    """
    if torch.unique(labels).shape[0] != 2:
        raise ValueError("Labels must be binary")

    unique_labels = torch.unique(labels)
    label_mapping = {unique_labels[0].item(): -1, unique_labels[1].item(): 1}
    return torch.tensor([label_mapping[label.item()] for label in labels])


def get_stable_corr_mat(data: np.array) -> np.array:
    """
    Calculate and return the stable correlation matrix for the given data.

    The function computes the absolute value of the covariance and correlation matrices,
    handles the cases where standard deviation is zero, and replaces NaNs with zeros.

    :param data: A 2D numpy array with each row representing a feature and each column representing an observation.
    :return: A 2D numpy array representing the stable correlation matrix.
    """

    # Calculate the absolute value of the covariance matrix.
    cov_mat = np.abs(np.cov(data, rowvar=False))

    # Calculate the absolute value of the variance vector from the covariance matrix.
    var_vec = np.abs(np.diag(cov_mat))

    # Calculate the standard deviation vector from the variance vector.
    stddev_vec = np.sqrt(var_vec)

    # Find the indices with zero standard deviation.
    zero_std_indices = [i for i, stddev in enumerate(stddev_vec) if stddev == 0]

    # Calculate the absolute value of the correlation matrix.
    corr_mat = np.abs(np.corrcoef(data, rowvar=False))

    # Replace all NaNs with 0.
    corr_mat = np.nan_to_num(corr_mat)

    # Find the minimum non-zero element in the corr_mat.
    min_val = np.min(corr_mat.flatten()[np.nonzero(corr_mat.flatten())]) / 10.0

    # Replace rows and columns with zero standard deviation with min_val.
    corr_mat[zero_std_indices, :] = min_val
    corr_mat[:, zero_std_indices] = min_val

    return corr_mat


def get_sparse_corr_matrix(corr_matrix: np.array, cache_directory: str) -> np.array:
    """
    Computes a sparse version of the input correlation matrix using graph sparsification.

    This function saves the input correlation matrix as a .npy file in the specified cache directory,
    then runs a Julia script to perform graph sparsification. The sparsified correlation matrix is
    loaded and returned as a NumPy array.

    Args:
        corr_matrix (np.array): The input correlation matrix to be sparsified.
        cache_directory (str): The directory where intermediate files will be saved and loaded.

    Returns:
        np.array: The sparsified correlation matrix.
    """
    save_name = "corr_matrix.npy"
    save_path = os.path.join(cache_directory, save_name)
    np.save(save_path, corr_matrix)

    graph_sparsification_path = "experiments_reproduction/graph_sparsification.jl"

    subprocess.run(["julia", graph_sparsification_path, save_path, cache_directory])

    sparse_load_path = os.path.join(cache_directory, "corr_matrix_sparsified_epsilon_0.15.npz")

    return np.load(sparse_load_path)
