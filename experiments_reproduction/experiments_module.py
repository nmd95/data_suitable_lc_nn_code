from abc import ABC
from typing import List, Union, Dict, Optional
import networkx as nx
import random
import torch
import torch.nn as nn
import numpy as np
import os
from rearrangement_algorithms.igtd import IGTD
from rearrangement_algorithms.rearrangement_alg import rearrange_from_data
from entropy.entropy_module import EntropyCalc
from common.data_utils import get_stable_corr_mat, permute_elements, rearrange_partitions
from models.utils import get_model_and_criterion, get_optimizer_and_scheduler, prepare_model
from common.train_eval import Trainer


class Result(ABC):
    def __init__(self, context: str):
        self.context = context


class EntropyMeasurementResult(Result):
    def __init__(self, context: str, measurement_params: dict):
        super().__init__(context)
        self.measurement_params = measurement_params


class AveCanonicalResult(EntropyMeasurementResult):
    def __init__(self, entropy_at_level: dict, **kwargs):
        super().__init__(**kwargs)
        self.entropy_at_level = entropy_at_level


class AveSurrogateCanonicalResult(EntropyMeasurementResult):
    def __init__(self, entropy_at_level: dict, **kwargs):
        super().__init__(**kwargs)
        self.entropy_at_level = entropy_at_level


class RearrangementResult(Result):
    def __init__(self, context: str, permutation: List[int]):
        super().__init__(context)
        self.permutation = permutation

class CorrCutsResult(RearrangementResult):
    def __init__(self, cut_alg: str, cut_options: dict, **kwargs):
        super().__init__(**kwargs)
        self.cut_alg = cut_alg
        self.cut_options = cut_options


class IGTDResult(RearrangementResult):
    def __init__(self, delta_curve: List[float], no_imp_count_threshold: int,
                 no_imp_val_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.delta_curve = delta_curve
        self.num_iters = len(delta_curve)
        self.no_imp_count_threshold = no_imp_count_threshold
        self.no_imp_val_threshold = no_imp_val_threshold


class ModelAccuracyResult(Result):
    def __init__(self, context: str, test_acc: float, train_acc: float,
                 model_dir: Optional[str], val_acc: Union[float, None] = None, train_curve: Union[List[float], None] = None
                 ,val_curve: Union[List[float], None] = None):
        super().__init__(context)
        self.test_acc = test_acc
        self.val_acc = val_acc
        self.train_acc = train_acc
        self.model_dir = model_dir
        self.train_curve = train_curve
        self.val_curve = val_curve


class Experiment:
    # 'seed' is the random seed for producing random permutations
    # 'dim' is the number of features in the 1D-array
    def __init__(self, seed: int, dim: int, data_dir_path: str, num_folds=None, data_dims=None) -> None:
        self.dim = dim
        self.graph = self._init_graph()
        self.seed = seed
        self.random_generator = random.Random(seed)
        self.data_dir_path = data_dir_path
        self.data_dims = data_dims
        self.data_mods = None if data_dims is None else len(data_dims)
        if num_folds is None:
            self.folded = False
        else:
            self.folded = True
            self._num_folds = num_folds
        self.nodes_name_map = {}

    def _init_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node(0, node_role='control')
        attributes = {
            0: {
                'baseline_results': {},
                'entropy_measurement_results': {},
                'permutation': list(range(self.dim)),
            }
        }
        nx.set_node_attributes(graph, attributes)

        return graph
    def get_node_role(self, node: int) -> str:
        return self.graph.nodes[node]['node_role']

    def _get_data(self, data_name: str, load_path: str = None) -> np.ndarray:
        load_path = load_path or self.data_dir_path
        return np.load(f"{load_path}/{data_name}")

    def _get_random_data_sample_from_disk(self, data_name: str, labels_name: str,
                                          sample_size: int, load_path: str = None) -> np.ndarray:
        data_disk = self._get_data(data_name, load_path)
        num_samples = data_disk.shape[0]
        sample_size = min(sample_size, num_samples)

        shape = [sample_size] + list(self.data_dims)
        data_memory = np.zeros(shape)
        temp_l = list(range(num_samples))
        self.random_generator.shuffle(temp_l)

        sample_ind = temp_l[:sample_size]
        data_memory[:, :] = data_disk[sample_ind, :]
        data_memory = data_memory.astype(np.float64)

        labels = self._get_data(labels_name, load_path)
        labels = labels[sample_ind]
        return data_memory, labels

    def add_node(self, node_role, node_attrs, parent_node=None) -> int:
        new_node_index = self.graph.number_of_nodes()
        self.graph.add_node(new_node_index, node_role=node_role, **node_attrs)

        parent_node = parent_node if parent_node is not None else 0
        self.graph.add_edge(parent_node, new_node_index)
        return new_node_index

    def add_random_permutation(self, node=None) -> int:
        random_permutation = list(range(self.dim))
        self.random_generator.shuffle(random_permutation)
        attributes = {'permutation': random_permutation,
                      'baseline_results': {},
                      'entropy_measurement_results': {}}
        return self.add_node('random_permutation', attributes, node)

    def add_n_swaps_structured_permutation(self, n_swaps: int, node=None) -> int:
        split = list(range(self.dim))
        if n_swaps == -1:
            self.random_generator.shuffle(split)
        else:
            for _ in range(n_swaps):
                ind1, ind2 = self.random_generator.sample(range(self.dim), 2)
                split[ind1], split[ind2] = split[ind2], split[ind1]
        attributes = {'permutation': split,
                      'baseline_results': {},
                      'entropy_measurement_results': {},
                      'n_swaps': n_swaps}
        return self.add_node('n_swaps_structured_permutation', attributes, node)

    def add_rearrangement_permutation(self, random_permutation_node: int,
                                      rearrangement_result: RearrangementResult, fold=None) -> int:
        attributes = {'rearrangement_result': rearrangement_result,
                      'baseline_results': {},
                      'entropy_measurement_results': {},
                      'fold': fold}
        return self.add_node('rearrangement_permutation', attributes, random_permutation_node)

    def add_fold_rearrangement_permutation(self, random_permutation_node: int, fold_nodes: List[int]) -> int:
        attributes = {'fold_nodes': fold_nodes,
                      'baseline_results': {},
                      'entropy_measurement_results': {}}
        new_node_index = self.add_node('fold_rearrangement_permutation', attributes, random_permutation_node)
        for fold_node in fold_nodes:
            self.graph.add_edge(new_node_index, fold_node)
        return new_node_index

    # node is the node's number
    def add_baseline_result(self, node: int, result: ModelAccuracyResult, result_name: str) -> None:
        self.graph.nodes[node]['baseline_results'][result_name] = result

    # node is the node's number
    def add_entropy_measurement_result(self, node: int, result: EntropyMeasurementResult, result_name: str) -> None:
        self.graph.nodes[node]['entropy_measurement_results'][result_name] = result

    def get_permutation(self, node: int) -> List[int]:
        node_role = self.get_node_role(node)
        if node_role == 'rearrangement_permutation':
            rearrangement_result = self.get_rearrangement_result(node=node)
            return rearrangement_result.permutation

        else:
            try:
                temp = self.graph.nodes[node]['permutation']
            except KeyError as error:
                print("Error, this node doesn't have this attribute.")
                temp = None
            return temp

    def get_entropy_measurement_results(self, node: int) -> dict:
        try:
            temp = self.graph.nodes[node]['entropy_measurement_results']
        except KeyError as error:
            print("Error, this node doesn't have this attribute.")
            temp = None
        return temp

    def get_baseline_results(self, node: int) -> dict:
        try:
            temp = self.graph.nodes[node]['baseline_results']
        except KeyError as error:
            print("Error, this node doesn't have this attribute.")
            temp = None
        return temp

    def get_rearrangement_result(self, node: int) -> RearrangementResult:
        try:
            temp = self.graph.nodes[node]['rearrangement_result']
        except KeyError as error:
            print("Error, this node doesn't have this attribute.")
            temp = None
        return temp

    def _rearrange_node(self, node: int, rearrange_params: dict, fold=None) -> Union[int, None]:
        load_path = os.path.join(self.data_dir_path,
                                 f"fold_{str(fold)}") if self.folded and (fold is not None) else self.data_dir_path
        perm = self.get_permutation(node)

        data = self._get_data("train_data.npy", load_path=load_path)
        data = permute_elements(data, perm)

        sparsification_cache_dir = self.data_dir_path if rearrange_params.get("sparse", False) else None

        if rearrange_params["igtd"]:
            data = torch.from_numpy(data)
            igtd = IGTD(data_array=data, no_imp_counter=rearrange_params["no_imp_counter"],
                        no_imp_count_threshold=rearrange_params["no_imp_count_threshold"],
                        no_imp_val_threshold=rearrange_params["no_imp_val_threshold"],
                        device=rearrange_params["device"], sparse=rearrange_params.get("sparse", False),
                        cache_dir=sparsification_cache_dir)
            igtd.run(t=rearrange_params["t"])

            igtd_permutation = list(np.asarray(perm)[igtd.get_permutation()])

            igtd_res = IGTDResult(delta_curve=igtd.get_update_delta_history(),
                                  no_imp_count_threshold=rearrange_params["no_imp_count_threshold"],
                                  no_imp_val_threshold=rearrange_params["no_imp_val_threshold"], context="",
                                  permutation=igtd_permutation)
            del igtd

            return self.add_rearrangement_permutation(random_permutation_node=node,
                                                      rearrangement_result=igtd_res, fold=fold)

        else:

            cut_permutation = rearrange_from_data(data=data, cut_options=rearrange_params["cut_options"],
                                                  cache_dir=sparsification_cache_dir, sparse=rearrange_params.get("sparse", False))
            cut_permutation = list(np.asarray(perm)[cut_permutation])

            cuts_res = CorrCutsResult(cut_alg=rearrange_params["cut_alg"], cut_options=rearrange_params["cut_options"],
                                      context="", permutation=cut_permutation)

            return self.add_rearrangement_permutation(random_permutation_node=node,
                                                      rearrangement_result=cuts_res, fold=fold)


    def rearrange_node(self, node: int, rearrange_params: dict) -> Union[int, None]:

        if not self.folded:
            return self._rearrange_node(node=node, rearrange_params=rearrange_params, fold=None)
        rearrange_nodes = [
            self._rearrange_node(node=node, rearrange_params=rearrange_params, fold=fold)
            for fold in range(self._num_folds)
        ]
        # add fold_rearrangement_permutation to the node
        return self.add_fold_rearrangement_permutation(random_permutation_node=node, fold_nodes=rearrange_nodes)

    def entropy_measure_node(self, node: int, meas_params: dict, fold=None, surrogate=False) -> None:
        load_path = os.path.join(self.data_dir_path,
                                 f"fold_{str(fold)}") if self.folded and fold else self.data_dir_path
        perm = self.get_permutation(node)

        ec = EntropyCalc(device=meas_params["device"], theta=meas_params.get("theta"),
                         svd_options=meas_params.get("svd_options"))

        data, labels = self._get_random_data_sample_from_disk(data_name="train_data.npy", sample_size=meas_params["sample_size"],
                                                              labels_name="train_labels.npy", load_path=load_path)

        data, labels = rearrange_partitions(permute_elements(data, perm)), torch.from_numpy(labels)

        if surrogate:
            corr_mat = torch.from_numpy(get_stable_corr_mat(data)).to(meas_params["device"])
            entropies = ec.surrogate_canonical_at_levels(meas_params["levels"], corr_mat, self.data_mods)
            result = AveSurrogateCanonicalResult(entropy_at_level=entropies, context="", measurement_params=meas_params)
        else:
            data = torch.from_numpy(data).to(meas_params["device"])
            entropies = ec.canonical_at_levels(meas_params["levels"], meas_params["ee"], data, labels,
                                               meas_params["fb_size"], meas_params["batch_size"], meas_params["n_batches"],
                                               self.data_mods)
            result = AveCanonicalResult(entropy_at_level=entropies, context="", measurement_params=meas_params)

        self.add_entropy_measurement_result(node=node, result=result, result_name=meas_params["run_name"])
        return None

    # get all nodes with given role
    def get_nodes_by_role(self, role: str) -> List[int]:
        return [node for node in range(self.graph.number_of_nodes()) if self.get_node_role(node) == role]

    def get_n_swaps_nodes(self, n_swaps: int) -> List[int]:
        n_swaps_structured_permutation_nodes = self.get_nodes_by_role("n_swaps_structured_permutation")
        return [node for node in n_swaps_structured_permutation_nodes if self.graph.nodes[node]["n_swaps"] == n_swaps]

    def _make_baseline_result(self, node: int, baseline_params: dict, data_dir: str):
        perm = self.get_permutation(node)
        if perm is None:
            print("Error: perm is None")
            return None

        model, criterion = get_model_and_criterion(baseline_params)
        model = prepare_model(model, baseline_params)

        if baseline_params["model_name"] == "S4Model":
            optimizer, scheduler = model.setup_optimizer(model, **baseline_params["s4_setup_params"])
            if baseline_params["multiple_gpu"]:
                model = nn.DataParallel(model, device_ids=baseline_params["device_ids"])
        else:
            optimizer, scheduler = get_optimizer_and_scheduler(model, baseline_params)

        t = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_classes=baseline_params["num_classes"], permutation=perm, device=baseline_params["device"],
                    weights_save_path=baseline_params["weights_save_path"],
                    data_dims=self.data_dims, seed=self.seed, data_dir=data_dir,
                    batch_sizes=baseline_params["batch_sizes"])

        t.train_val_test(epochs=baseline_params["epochs"])

        final_results = t.get_final_results()

        return ModelAccuracyResult(train_curve=final_results["epoch_train_acc"],
                                   val_curve=final_results["epoch_val_acc"],
                                   test_acc=final_results["best_model_test_acc"],
                                   train_acc=final_results["best_model_train_acc"],
                                   model_dir=final_results["weights_save_path"], context="",
                                   val_acc=final_results["best_val_acc"])

    def _run_baseline_node(self, node: int, baseline_params: dict, folded: bool = False):
        if not folded:
            data_dir = self.data_dir_path
            result = self._make_baseline_result(node, baseline_params, data_dir)
        else:
            result = self._treat_folded_case(node, baseline_params)

        self.add_baseline_result(node=node, result=result, result_name=baseline_params["run_name"])

    def _treat_folded_case(self, node: int, baseline_params: dict):
        if self.get_node_role(node) == "fold_rearrangement_permutation":
            nodes_pointed_to_by_node = self.graph.successors(node)
            baseline_results = []
            for node_pointed_to_by_node in nodes_pointed_to_by_node:
                fold = self.graph.nodes[node_pointed_to_by_node]["fold"]
                data_dir = os.path.join(self.data_dir_path, f"fold_{fold}")
                result = self._make_baseline_result(node_pointed_to_by_node, baseline_params, data_dir=data_dir)
                baseline_results.append(result)
            test_accs = [baseline_result.test_acc for baseline_result in baseline_results]
            train_accs = [baseline_result.train_acc for baseline_result in baseline_results]

            avg_test_acc = sum(test_accs) / len(test_accs)
            avg_train_acc = sum(train_accs) / len(train_accs)
            result = ModelAccuracyResult(train_acc=avg_train_acc, test_acc=avg_test_acc, context="", model_dir=None)
        else:
            results = []
            for fold in range(self._num_folds):
                data_dir = os.path.join(self.data_dir_path, f"fold_{fold}")
                result = self._make_baseline_result(node, baseline_params, data_dir)
                results.append(result)
            test_accs = [result.test_acc for result in results]
            train_accs = [result.train_acc for result in results]

            test_acc = np.mean(test_accs)
            train_acc = np.mean(train_accs)

            result = ModelAccuracyResult(train_acc=train_acc, test_acc=test_acc, context="", model_dir=None)
        return result

    def run_baseline_node(self, node: int, baseline_params: dict):
        if self.folded:
            self._run_baseline_node(node, baseline_params, folded=True)
        else:
            self._run_baseline_node(node, baseline_params)








































