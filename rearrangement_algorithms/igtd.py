import numpy as np
import torch
from collections import deque
from common.data_utils import create_index_map, get_stable_corr_mat, rearrange_partitions, get_sparse_corr_matrix
from typing import List, Tuple, Optional



def create_q_multidim(index_mapping: np.ndarray) -> np.ndarray:
    """
    Create a Q matrix representing the distance between each pair of elements in the given index mapping.

    Args:
        index_mapping (np.ndarray): The index mapping generated using the `create_index_map` function.

    Returns:
        np.ndarray: The Q matrix containing the distances between each pair of elements in the index mapping.
    """
    shape = index_mapping.shape
    num_entries = np.prod(shape)
    q = np.zeros((num_entries, num_entries))

    # Calculate the pairwise distances between elements and fill the Q matrix
    for i in range(num_entries):
        for j in range(num_entries):
            i_index = np.argwhere(
                index_mapping == i
            )  # N-dimensional index of the value i in index_mapping
            j_index = np.argwhere(
                index_mapping == j
            )  # N-dimensional index of the value j in index_mapping
            dist = np.linalg.norm(
                i_index - j_index
            )  # Calculate the distance between the two indices
            q[i, j] = dist

    return q


class IGTD:
    def __init__(
        self,
        data_array: torch.tensor,
        no_imp_counter: int = 0,
        no_imp_count_threshold: int = 5,
        no_imp_val_threshold: float = 0.0,
        device: str = "cuda:0",
        sparse: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the IGTD class with input data and parameters.

        Args:
            data_array (torch.tensor): Input data tensor. Assuming the first dimension is the batch dimension (greater at least 1).
            no_imp_counter (int, optional): Number of iterations without improvement counter. Default is 0.
            no_imp_count_threshold (int, optional): Threshold for the number of iterations without improvement. Default is 5.
            no_imp_val_threshold (float, optional): Threshold for the improvement value. Default is 0.0.
            device (str, optional): Device for torch tensors. Default is "cuda:0".
            sparse (bool, optional): Whether to use a sparse correlation matrix. Default is False.
            cache_dir (Optional[str], optional): Directory to store the cached correlation matrix. Default is None.
        """
        self._device = device
        self._sparse = sparse
        self._cache_dir = cache_dir
        self._data_dims = data_array.shape[1:]
        self._data_mods = len(data_array.shape[1:])
        self._data_array = rearrange_partitions(data_array)
        self._num_features = self._data_array.shape[-1]
        self._permutation = list(range(self._num_features))
        self._recency_stack = self._initialize_stack()
        self._q = self._initialize_q()
        self._r_init = self._initialize_r()
        self._r_current = self._r_init.clone()
        self.no_imp_counter = no_imp_counter
        self._no_imp_count_threshold = no_imp_count_threshold
        self._no_imp_val_threshold = no_imp_val_threshold
        self._update_delta_history = []

    def _initialize_r(self) -> torch.tensor:
        """
        Initialize the R matrix, which represents the rank-based structure of the correlation matrix.
        The R matrix is created by taking the correlation matrix of the data and replacing the
        correlation values with their ranks in ascending order along the rows.

        Returns:
            torch.tensor: The R matrix with rank-based structure of the correlation matrix.
        """
        # Get the stable correlation matrix from the data array
        corr_mat = get_stable_corr_mat(self._data_array.numpy())
        if self._sparse:
            corr_mat = get_sparse_corr_matrix(corr_mat, self._cache_dir)

        r_orig = torch.from_numpy(corr_mat)

        r = torch.zeros((self._num_features, self._num_features))

        # Replace correlation values with their ranks in ascending order along the rows
        for i in range(self._num_features):
            row = r_orig[i, :]
            ascending_ranks = torch.argsort(row, descending=True)
            ascending_ranks_fixed = torch.tensor(
                list(range(self._num_features))
            ).float()
            r[i, ascending_ranks] = ascending_ranks_fixed

        r = r.to(self._device)
        return r

    def _initialize_q(self) -> torch.tensor:
        """
        Initialize the Q matrix, which represents the coordinate distance between each pair of elements in the given index mapping.
        The Q matrix is created based on the index mapping generated using the `create_index_map` function.

        Returns:
            torch.tensor: The Q matrix containing the distances between each pair of elements in the index mapping.
        """
        # Create Q matrix based on the index mapping
        index_mapping = create_index_map(self._data_dims)
        q = create_q_multidim(index_mapping)
        q = torch.tensor(q)
        q = q.to(self._device)
        return q

    def _initialize_stack(self) -> deque:
        """
        Initialize the recency stack with the initial permutation.

        Returns:
            deque: The initialized recency stack containing the initial permutation.
        """
        return deque(self._permutation)

    def _get_update_index(self) -> int:
        """
        Get the index of the element to be updated in the current permutation.
        The update index corresponds to the least recently updated element.

        Returns:
            int: The index of the element to be updated in the current permutation.
        """
        recency_index = self._recency_stack.pop()
        self._recency_stack.append(recency_index)
        return self._permutation.index(recency_index)

    def _update_recency_stack(self) -> None:
        """
        Update the recency stack by moving the least recently updated element to the most recently updated position.
        This function is called after a successful swap operation.
        """
        recency_index = self._recency_stack.pop()
        self._recency_stack.appendleft(recency_index)

    def _update_permutation(self, i: int, j: int) -> None:
        """
        Update the permutation list by swapping the elements at the given indices.

        Args:
            i (int): The index of the first element to be swapped in the permutation.
            j (int): The index of the second element to be swapped in the permutation.
        """
        ind_i, ind_j = self._permutation[i], self._permutation[j]
        self._permutation[i] = ind_j
        self._permutation[j] = ind_i

    def _swap_in_r(self, i: int, j: int) -> None:
        """
        Swap rows and columns i and j in the R matrix.

        Args:
            i (int): The index of the first row/column to be swapped.
            j (int): The index of the second row/column to be swapped.
        """
        # Swap rows i and j
        r_i, r_j = self._r_current[i, :].clone(), self._r_current[j, :].clone()
        r_i, r_j = r_i.to(self._device), r_j.to(self._device)
        self._r_current[i, :] = r_j
        self._r_current[j, :] = r_i

        # Swap columns i and j
        c_i, c_j = self._r_current[:, i].clone(), self._r_current[:, j].clone()
        c_i, c_j = c_i.to(self._device), c_j.to(self._device)
        self._r_current[:, i] = c_j
        self._r_current[:, j] = c_i

    def _get_q_r_dist(self) -> torch.tensor:
        """
        Calculate the sum of squared differences between the Q and R matrices.

        Returns:
            torch.tensor: The sum of squared differences between the Q and R matrices.
        """
        return torch.sum(torch.pow(self._q - self._r_current, 2))

    def _find_max_swap_ind(self, swap_ind: int) -> Tuple[int, float]:
        """
        Find the index that maximizes the decrease in Q-R distance when swapped with the current index.

        Args:
            swap_ind (int): The index of the current element in the permutation.

        Returns:
            Tuple[int, float]: The index that maximizes the decrease in Q-R distance and the corresponding delta value.
        """
        current_q_r_dist = self._get_q_r_dist()
        max_ind = swap_ind
        max_delta = 0

        # Iterate through all indices and find the one with the maximum decrease in Q-R distance
        for i, _ in enumerate(self._permutation):
            self._swap_in_r(swap_ind, i)
            temp_delta = self._get_q_r_dist() - current_q_r_dist
            self._swap_in_r(swap_ind, i)

            if temp_delta < max_delta:
                max_delta = temp_delta
                max_ind = i

        return max_ind, max_delta

    def _swap_update(self, i: int, j: int) -> None:
        """
        Perform a swap update, which includes swapping rows and columns in R matrix, updating the permutation,
        and updating the recency stack.

        Args:
            i (int): The index of the first element to be swapped.
            j (int): The index of the second element to be swapped.
        """
        self._swap_in_r(i, j)  # Swap rows and columns i and j in R matrix
        self._update_permutation(i, j)  # Update the permutation
        self._update_recency_stack()  # Update the recency stack

    def _alg_step(self) -> str:
        """
        Perform a single step of IGTD and return the status of the algorithm.

        Returns:
            str: The status of the algorithm, either "continue" or "terminate".
        """
        # Get the index to update from the recency stack
        swap_ind = self._get_update_index()

        # Find the index with the maximum delta value when swapped with swap_ind
        max_ind, max_delta = self._find_max_swap_ind(swap_ind)

        # Update the delta history with the negative of max_delta
        self._update_delta_history.append(-1 * max_delta)

        # Check termination conditions
        if max_delta > -self._no_imp_val_threshold:
            self.no_imp_counter += 1
            if self.no_imp_counter >= self._no_imp_count_threshold:
                return "terminate"
        else:
            self.no_imp_counter = 0
            # Perform the swap update
            self._swap_update(swap_ind, max_ind)
            return "continue"

    def run(self, t: int) -> None:
        """
        Run the IGTD for a specified number of iterations or until
        the termination condition is met.

        Args:
            t (int): The maximum number of iterations to run the algorithm.
        """
        for _ in range(t):
            status = self._alg_step()
            if status == "terminate":
                print("tolerance termination")
                break

    def get_update_delta_history(self) -> List[float]:
        """
        Get a copy of the update delta history.

        Returns:
            List[float]: The update delta history.
        """
        return self._update_delta_history.copy()

    def get_q(self) -> torch.tensor:
        """
        Get a copy of the current Q matrix.

        Returns:
            torch.tensor: The current Q matrix.
        """
        return self._q.clone()

    def get_r_init(self) -> torch.tensor:
        """
        Get a copy of the initial R matrix.

        Returns:
            torch.tensor: The initial R matrix.
        """
        return self._r_init.clone()

    def get_r_current(self) -> torch.tensor:
        """
        Get a copy of the current R matrix.

        Returns:
            torch.tensor: The current R matrix.
        """
        return self._r_current.clone()

    def get_permutation(self) -> List[int]:
        """
        Get a copy of the current permutation.

        Returns:
            List[int]: The current permutation.
        """
        return self._permutation.copy()
