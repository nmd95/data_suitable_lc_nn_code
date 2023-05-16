import gc
import networkx as nx
import numpy as np
import nxmetis
from collections import deque
from common.data_utils import get_stable_corr_mat, rearrange_partitions, get_sparse_corr_matrix
from typing import Dict, List, Union, Optional


class TreeNode:
    def __init__(self):
        self.children = []
        self.data = None


def partition_graph(
    graph: nx.Graph, cut_options: Dict[str, Union[int, bool]]
) -> List[List[int]]:
    """
    Partition the input graph into subgraphs based on the cut_options provided.

    Args:
        graph (nx.Graph): The input graph to be partitioned.
        cut_options (Dict[str, Union[int, bool]]): A dictionary containing partitioning options.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains the node indices for a partition.
    """

    # Set up Metis options for partitioning
    metis_options = nxmetis.MetisOptions(
        niter=cut_options["niter_metis"],
        ncuts=cut_options["ncuts_metis"],
        seed=cut_options["seed_metis"],
        objtype=nxmetis.enums.MetisObjType.cut,
    )

    num_parts = cut_options.get("num_parts", 2)  # default to 2

    # Partition the graph using nxmetis
    _, partitions = nxmetis.partition(
        graph,
        num_parts,
        edge_weight="weight",
        options=metis_options,
        recursive=cut_options["metis_recursive"],
    )

    return [list(part) for part in partitions]


def rearrange_from_corr_matrix(
    corr_mat: np.array, cut_options: Dict[str, Union[int, bool]]
) -> List[int]:
    """
    Rearrange the order of indices based on the input correlation matrix.

    Args:
        corr_mat (np.array): The input correlation matrix.
        cut_options (Dict[str, Union[int, bool]]): A dictionary containing partitioning options.

    Returns:
        List[int]: A list of rearranged indices.
    """

    num_parts = cut_options["num_parts"]

    # Build the graph using the correlation matrix
    graph = nx.Graph()
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            if i != j:
                val = corr_mat[i, j]
                weight = round(val * 1e10)
                graph.add_edge(i, j, weight=weight)

    # Initialize the root TreeNode with the graph nodes
    root = TreeNode()
    root.data = list(graph.nodes())

    queue = deque([root])

    # Breadth-first search (BFS) to partition the graph
    while queue:
        current_node = queue.pop()

        # Stop partitioning when the number of nodes is below a certain threshold
        if len(current_node.data) <= (num_parts + 1):
            continue

        subgraph = graph.subgraph(current_node.data)
        partitions = partition_graph(subgraph, cut_options)

        # Create child TreeNodes for each partition
        for part in partitions:
            child = TreeNode()
            child.data = part
            current_node.children.append(child)
            queue.appendleft(child)

        # Clean up memory
        del subgraph
        gc.collect()

    return _dfs_leaf_read(root, num_parts)


def _dfs_leaf_read(tree_root: TreeNode, num_parts: int) -> List[int]:
    """
    Perform a depth-first search on the tree rooted at `tree_root` to collect
    the rearranged order of indices from the leaf nodes with a maximum of
    `num_parts + 1` elements.

    Args:
        tree_root (TreeNode): The root node of the tree to traverse.
        num_parts (int): The number of partitions used in graph partitioning.

    Returns:
        List[int]: The rearranged order of indices.
    """
    final_rearrangement = []

    def _recursive_children_stack(stack: deque, node: TreeNode) -> None:
        """
        Recursively traverse the tree and store nodes in a depth-first order.

        Args:
            stack (deque): A deque to store nodes in depth-first order.
            node (TreeNode): The current node being traversed.
        """
        if node.data is not None:
            stack.append(node.data)

        for child in node.children:
            _recursive_children_stack(stack, child)

    queue = deque()
    _recursive_children_stack(queue, tree_root)

    # Process nodes in depth-first order
    while queue:
        current_node = queue.popleft()
        # Collect nodes with a maximum of (num_parts + 1) elements
        if len(current_node) <= (num_parts + 1):
            final_rearrangement.extend(current_node)

    return final_rearrangement


def rearrange_from_data(
    data: np.array, cut_options: Dict[str, Union[int, bool]], cache_dir: Optional[str] = None, sparse: bool = False
) -> List[int]:
    """
    Rearrange the features of `data` based on graph partitioning of a correlation matrix
    computed from the input data. This function first rearranges the data into partitions,
    computes a stable correlation matrix from the data, and then optionally sparsifies
    the correlation matrix using the `get_sparse_corr_matrix` function. Finally, it
    rearranges the rows based on the correlation matrix and graph partitioning.

    Args:
        data (np.array): The input data as a NumPy array.
        cut_options (Dict[str, Union[int, bool]]): A dictionary of options for
            the graph partitioning algorithm.
        cache_dir (Optional[str]): The directory where intermediate files will be saved
            and loaded during the computation of the sparse correlation matrix. Only
            used if `sparse` is set to True. Default is None.
        sparse (bool): If set to True, the correlation matrix will be sparsified using
            the `get_sparse_corr_matrix` function before rearranging the rows.
            Default is False.

    Returns:
        List[int]: The rearranged order of indices.
    """
    # Rearrange data into partitions
    data = rearrange_partitions(data)
    # Compute a stable correlation matrix
    corr_mat = get_stable_corr_mat(data)
    if sparse:
        corr_mat = get_sparse_corr_matrix(corr_mat, cache_dir)
    # Rearrange rows based on the correlation matrix and graph partitioning
    return rearrange_from_corr_matrix(corr_mat, cut_options)

