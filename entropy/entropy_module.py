import math
import numpy as np
from typing import List, Tuple, Optional
import torch


class EntropyCalc:
    def __init__(self, device: Optional[str], theta: Optional[float], svd_options: Optional[dict]) -> None:
        """
        Initialize the EntropyCalc class with required parameters.

        Args:
            device (str): Device to use for torch tensors (e.g. 'cuda:0' or 'cpu')
            theta (float): Theta parameter for the angle of embedding (as a fraction of pi).
            svd_options (dict): Dictionary containing options for SVD
        """
        self._device = device
        self._PI = torch.tensor(math.pi).to(device).type(torch.float64) if device is not None else None
        self._theta = torch.tensor(theta).to(device).type(torch.float64) if theta is not None else None
        self._svd_options = svd_options if svd_options is not None else None

    def _get_gm_sub_features(
        self, data: torch.tensor, f_inds: List[int]
    ) -> torch.tensor:
        """
        Compute the Gram matrix for the given data and a subst of feature indices.

        Args:
            data (torch.tensor): Input data tensor
            f_inds (List[int]): List of feature indices

        Returns:
            torch.tensor: Gram matrix
        """
        m = data.shape[0]
        data_temp = data[:, f_inds]
        t_1 = data_temp.unsqueeze(0).expand(m, -1, -1)
        t_2 = t_1.transpose(1, 0)
        t_3 = torch.cos(self._PI * self._theta * (t_1 - t_2))
        return torch.prod(t_3, dim=2)

    def _get_gm_batch(
        self, data: torch.tensor, part: List[int], fb_size: int
    ) -> torch.tensor:
        """
        Compute the Gram matrix batch for the given data and partition.

        Args:
            data (torch.tensor): Input data tensor
            part (List[int]): List of feature indices for the partition
            fb_size (int): Feature batch size

        Note:
            This function is used to compute the Gram matrix for a partition of features in batches over the features
            in the partition. This is done to avoid memory issues when computing the Gram matrix for large partitions.

        Returns:
            torch.tensor: Gram matrix batch
        """
        m = data.shape[0]
        n_f = len(part)

        # Compute batch sizes
        batch_sizes = [fb_size] * (n_f // fb_size)
        rem = n_f % fb_size
        if rem > 0:
            batch_sizes += [rem]

        # Convert to numpy array and create batches
        part_np = np.array(part)
        batches = [
            part_np[offset : offset + bs].tolist()
            for offset, bs in enumerate(batch_sizes)
        ]

        data = data.type(torch.float64)
        running_g = torch.ones((m, m)).to(self._device)

        for b in batches:
            running_g *= self._get_gm_sub_features(data, b)
            torch.cuda.empty_cache()

        return running_g

    def _get_ent_from_gm(
        self, g_a: torch.tensor, g_b: torch.tensor
    ) -> Tuple[float, float]:
        """
        Calculate entanglement entropy and geometric entanglement from the two input Gram matrices.

        Args:
            g_a (torch.tensor): First input Gram matrix (for partition A)
            g_b (torch.tensor): Second input Gram matrix (for partition B - the complement of A)

        Returns:
            Tuple[float, float]: Entanglement entropy and geometric entanglement
        """

        def _vidal(m):
            m_prime = m + self._svd_options["vidal_noise"] * torch.mean(
                m
            ) * torch.rand_like(m).to(self._device)
            a, b, _ = torch.linalg.svd(m_prime, full_matrices=False)
            valid_cols = b != 0
            b = b[valid_cols]
            a = a[:, valid_cols].T
            return torch.matmul(torch.diag(b**0.5), a)

        v_a, v_b = _vidal(g_a), _vidal(g_b)
        f = torch.matmul(v_a, v_b.T)
        u, d, v = torch.linalg.svd(f, full_matrices=False)

        norm = torch.norm(d)
        spectrum = d / norm

        c = (spectrum**2) / torch.sum(spectrum**2)
        ge_val = (1 - torch.max(c)) ** 0.5
        ee_val = torch.distributions.Categorical(probs=c).entropy()

        torch.cuda.empty_cache()
        return ee_val.item(), ge_val.item()

    def _entropies_from_partition(
        self,
        data: torch.tensor,
        labels: Optional[torch.tensor],
        a_part: list,
        b_part: list,
        fb_size: int,
        batch_size: int,
        n_batches: int,
    ) -> Tuple[float, float]:
        """
        Calculate entanglement entropy and geometric entanglement for the given partition.

        Args:
            data (torch.tensor): Input data tensor
            labels (Optional[torch.tensor]): Optional tensor containing labels
            a_part (list): List of feature indices for partition A
            b_part (list): List of feature indices for partition B
            fb_size (int): Feature batch size
            batch_size (int): Number of random samples for each batch
            n_batches (int): Number of batches

        Returns:
            Tuple[float, float]: Average entanglement entropy and geometric entanglement
        """
        ave_ee_val, ave_ge_val = 0, 0

        for _ in range(n_batches):
            # Sample batch_size random samples from x_fs
            batch_inds = np.random.choice(data.shape[0], batch_size, replace=False)
            data_batch = data[batch_inds, :]

            # If labels are provided, sample the same batch of labels
            labels_batch = labels[batch_inds] if labels is not None else None

            gm_a = self._get_gm_batch(data_batch, a_part, fb_size)
            gm_b = self._get_gm_batch(data_batch, b_part, fb_size)

            if labels is not None:
                labels_batch = labels_batch.to(self._device)
                labels_batch = torch.where(labels_batch == 0, -1, 1)
                gm_sign = torch.outer(labels_batch, labels_batch)
                gm_a = gm_a * gm_sign

            temp_ee_val, temp_ge_val = self._get_ent_from_gm(gm_a, gm_b)

            ave_ee_val += temp_ee_val
            ave_ge_val += temp_ge_val

        ave_ee_val /= n_batches
        ave_ge_val /= n_batches

        return ave_ee_val, ave_ge_val

    @staticmethod
    def _surrogate_entropies_from_partition(
        corr_mat: torch.tensor, a_part: list, b_part: list, normalized=False
    ) -> float:
        """
        Compute the surrogate entropy for a given partition.

        This function calculates the surrogate entropy based on the correlation matrix.
        The surrogate entropy is the sum of the correlation matrix elements for a given partition.
        Optionally, it can return the normalized surrogate entropy by dividing the sum by the number of non-zero elements.

        Args:
            corr_mat (torch.tensor): The correlation matrix of shape (n_features, n_features).
            a_part (list): A list of integers representing the indices of the first partition.
            b_part (list): A list of integers representing the indices of the second partition.
            normalized (bool, optional): If True, return the normalized surrogate entropy. Defaults to False.

        Returns:
            float: The surrogate entropy (or normalized surrogate entropy) for the given partition.
        """

        # Get the sub correlation matrix corresponding to the given partition
        sub_corr_mat = corr_mat[a_part, :][:, b_part]

        # Calculate the cut value by summing the elements of the sub correlation matrix
        cut_value = torch.sum(sub_corr_mat)

        # Find the number of non-zero elements in the sub correlation matrix
        non_zero = torch.nonzero(sub_corr_mat)
        cut_size = non_zero.shape[0]

        # Calculate the (normalized) surrogate entropy
        surrogate_entropy = cut_value / cut_size if normalized else cut_value

        return surrogate_entropy.item()

    @staticmethod
    def _get_all_partitions_at_level(
        dim: int, level: int, data_mods: int
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Get all canonical partitions at a specified level of the tensor network tree.

        Parameters:
        dim (int): The feature space dimension (the total number of features).
        level (int): The level in the tensor network tree.
        data_mods (int): The number of axis (mods) of the data tensor (e.g. 1 for a vector, 2 for a matrix).

        Returns:
        List[Tuple[List[int], List[int]]]: A list of tuples, where each tuple contains two lists
                                          representing the indices of the features in the partition
                                          and its complement.
        """
        partitions = []
        level_zero = np.asarray(list(range(dim)))

        # Calculate the number of chunks at the given level
        num_chunks = int((2**data_mods) ** level)

        # Split the level_zero array into num_chunks equal parts
        basic_chunks = np.array_split(level_zero, num_chunks)
        basic_chunks = [list(arr) for arr in basic_chunks]

        # If the level and data_mods are both 1, return the basic partition
        if (level == 1) and (data_mods == 1):
            return [(basic_chunks[0], basic_chunks[1])]

        # For each basic_chunk, create a partition with its complement
        for basic_chunk in basic_chunks:
            mask = np.ones(len(level_zero), dtype=bool)
            mask[basic_chunk] = False
            complement_chunk = level_zero[mask]
            complement_chunk = list(complement_chunk)
            partitions.append((basic_chunk, complement_chunk))

        return partitions

    def canonical_at_levels(
        self,
        levels: List[int],
        ee: bool,
        data: torch.tensor,
        labels: Optional[torch.tensor],
        fb_size: int,
        batch_size: int,
        n_batches: int,
        data_mods: int,
    ) -> dict:
        """
        Calculate average entropies for given levels of partitions.

        Args:
            levels (List[int]): List of levels of partitions to calculate the average entropies.
            ee (bool): If True, calculate average entanglement entropy; otherwise, calculate average geometric entropy.
            data (torch.tensor): Data tensor of shape (num_samples, num_features).
            labels (Optional[torch.tensor]): Optional tensor containing labels of the samples. If provided, they will be treated as an additional feature.
            fb_size (int): Size of the feature batches.
            batch_size (int): Size of the data batches.
            n_batches (int): Number of data batches to use for calculating average entropy.
            data_mods (int): The number of axis (mods) of the data tensor (e.g. 1 for a vector, 2 for a matrix).

        Returns:
            dict: A dictionary containing the average entropies at each level, with keys in the format "level_i".
        """
        # Move data to the specified device
        data = data.to(self._device)

        # Initialize an empty dictionary to store average entropies per level
        per_level_entropies = {}

        # Iterate through the given levels
        for level in levels:
            # Get all possible partitions for the current level
            level_partitions = self._get_all_partitions_at_level(
                dim=data.shape[1], level=level, data_mods=data_mods
            )

            # Calculate the average entropy over all partitions at the current level
            level_ave_entropy = self._ave_entropy_over_partitions(
                partitions=level_partitions,
                data=data,
                labels=labels,
                corr_mat=None,
                fb_size=fb_size,
                batch_size=batch_size,
                n_batches=n_batches,
                surrogate=False,
                ee=ee,
            )

            # Add the calculated average entropy to the dictionary
            per_level_entropies[f"level_{str(level)}"] = level_ave_entropy

        return per_level_entropies

    def surrogate_canonical_at_levels(
        self, levels: List[int], corr_mat: torch.tensor, data_mods: int
    ) -> dict:
        """
        Compute the surrogate canonical entropies for all partitions at the specified levels.

        Args:
            levels (List[int]): List of levels of partitions to calculate the average entropies.
            corr_mat (torch.tensor): The correlation matrix of the data.
            data_mods (int):The number of axis (mods) of the data tensor (e.g. 1 for a vector, 2 for a matrix).

        Returns:
            dict: A dictionary containing the average surrogate canonical entropies at each level.
        """

        per_level_entropies = (
            {}
        )  # A dictionary to store the average entropies per level.

        # Iterate over each specified level.
        for level in levels:
            # Get all possible partitions for the current level.
            level_partitions = self._get_all_partitions_at_level(
                dim=corr_mat.shape[0], level=level, data_mods=data_mods
            )

            # Compute the average entropy for the current level.
            level_ave_entropy = self._ave_entropy_over_partitions(
                partitions=level_partitions,
                data=None,
                labels=None,
                corr_mat=corr_mat,
                fb_size=None,
                batch_size=None,
                n_batches=None,
                surrogate=True,
                ee=None,
            )

            # Store the average entropy for the current level in the output dictionary.
            per_level_entropies[f"level_{str(level)}"] = level_ave_entropy

        return per_level_entropies

    def _ave_entropy_over_partitions(
        self,
        partitions: List[Tuple[List[int], List[int]]],
        data: Optional[torch.tensor],
        labels: Optional[torch.tensor],
        corr_mat: Optional[torch.tensor],
        fb_size: Optional[int],
        batch_size: Optional[int],
        n_batches: Optional[int],
        surrogate: bool = False,
        ee: Optional[bool] = True,
    ) -> float:
        """
        Compute the average entropy over the provided partitions.

        Args:
            partitions (List[Tuple[List[int], List[int]]]): A list of tuples containing two lists of partition indices.
            data (Optional[torch.tensor]): The input data tensor. Required if not using surrogate method.
            labels (Optional[torch.tensor]): The input labels tensor.
            corr_mat (Optional[torch.tensor]): The input correlation matrix tensor. Required if using surrogate method.
            fb_size (Optional[int]): The number of features in each batch. Required if not using surrogate method.
            batch_size (Optional[int]): The number of samples in each batch. Required if not using surrogate method.
            n_batches (Optional[int]): The number of batches. Required if not using surrogate method.
            surrogate (bool, optional): If True, use surrogate method for entropy calculation. Defaults to False.
            ee (Optional[bool], optional): If True, compute entanglement entropy, else compute geometric entanglement. Required if not using surrogate method.

        Returns:
            float: The average entropy across the provided partitions.
        """

        entropy_sum = 0.0
        for partition in partitions:
            # If using surrogate method, calculate surrogate entropy
            if surrogate:
                entropy_calc = self._surrogate_entropies_from_partition(
                    corr_mat=corr_mat, a_part=partition[0], b_part=partition[1]
                )
            # Otherwise, calculate entanglement or geometric entanglement
            else:
                entropy_calc = self._entropies_from_partition(
                    data=data,
                    labels=labels,
                    a_part=partition[0],
                    b_part=partition[1],
                    fb_size=fb_size,
                    batch_size=batch_size,
                    n_batches=n_batches,
                )[0 if ee else 1]
            entropy_sum += entropy_calc

        # Return the average entropy over all partitions
        return entropy_sum / len(partitions)
