# What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement (NeurIPS 2023)


Code implementation for the algorithms introduced in [What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement](https://arxiv.org/abs/2303.11249) (NeurIPS 2023).

## Configuration

This project uses YAML configuration files to set various parameters for the experiments. You can find the configuration files in the `configs` folder.


### Parameters

Here is a list of common parameters and their explanations:

- `data_dir_name`: The name of the directory containing the dataset. The directory should be located in the `user_datasets` folder. See for example the `cifar10` folder. 
- `restructure_method`: The restructuring method used in the experiment (e.g., `feature_rearrangement_algorithm`, `igtd`).
- `device`: The device ID to be used for computation (0 for the first GPU, -1 for CPU, etc.).
- `data_name`: The name of the array containing the data.
- `labels_name`: The name of the array containing the labels.

#### Feature Rearrangement algorithm Parameters

- `cut_method`: The graph partitioning method used (e.g., `metis`).
- `niter_metis`: Number of iterations for the METIS algorithm.
- `ncuts_metis`: Number of cuts for the METIS algorithm.
- `metis_recursive`: Boolean value specifying whether to use recursive METIS partitioning.
- `seed_metis`: Seed value for the METIS algorithm.
- `num_parts`: Number of parts in each partition.

#### IGTD Parameters

- `no_imp_counter`: Initial value for the counter of iterations with no improvement.
- `no_imp_count_threshold`: Threshold for the number of iterations without improvement.
- `no_imp_val_threshold`: Threshold for the value of improvement.
- `t`: Maximal of iterations.


#### Entropy Measurement Parameters

- `levels`: The levels at which the entropy is to be measured.
- `entropy`: The entropy measurement method used (e.g., `entanglement entropy`, `geometric entropy`, `surrogate entropy`).
- `theta`: The angle of embedding for the entanglement entropy measurements (specified as a fraction of pi).
- `n_batches`: The number of batches to be used for the entropy measurements (batches are averaged to get any single measurement)
- `bath_size`: The size of each batch to be used for the entropy measurements.
- `fb_size`: The size of the feature batch to be used for the entropy measurements (can affect the efficiency of the measurements; should be set lower than the number of features in the data). 
- `vidal_noise`: A small amount of noise added to stabilize the SVD (used for the surrogate entropy measurements). 
## Usage

For restructuring the data, follow the example notebook `example_usage_data_restructuring.ipynb`.
For conducting entropy measurements, follow the example of the notebook `example_usage_entropy_measurement.ipynb`.


## Reproducing Paper's Experiments

In order to reproduce the paper's experiments, use the provided bash script `run_experiment.sh`.


### Usage

Run the script with the following command line arguments:

```bash
./run_experiment.sh <device_numbers> <seeds> <script_id>
```
- `<device_numbers>`: Space-separated list of GPU device numbers to run the experiments on (e.g., "1 2 3 4 5").
- `<seeds>`: Space-separated list of seed values (e.g., "0 1 2 3 4").
- `<script_id>`: The script identifier corresponding to the experiment you want to run (e.g., "table_2_dna" or "figure_4").

For example, to run the `table_2_dna_multi_seed.py` experiment on devices 1 to 5 with seeds 0 to 4, execute the following command:

```bash
./run_experiment.sh "1 2 3 4 5" "0 1 2 3 4" "table_2_dna"
```
The available options for `<script_id>` are:

- `table_2_dna`
- `table_2_semeion`
- `table_2_isolet`
- `table_3`
- `table_1`
- `table_4`
- `figure_4` (also for running the experiment of Figure 5)
- `figure_8`


### Data for the Experiments
Prior to running the experiments, you should download the datasets used in the experiments and save them in the empty folder `experiments_reproduction/paper_datasets` The datasets can be downloaded from [this link](https://drive.google.com/drive/folders/198kRSBzlMS7QUxi60ZnJTb4XPTAPkGVU?usp=sharing).
Each experiment requires specific datasets as detailed below:
- `table_2_dna_multi_seed.py`: `dna` dataset.
- `table_2_semeion_multi_seed.py`: `semeion` dataset.
- `table_2_isolet_multi_seed.py`: `isolet` dataset.
- `table_3_multi_seed.py`: `cifar10` dataset.
- `table_1_multi_seed.py`: `speech_commands_dim_2048_downsized` dataset.
- `table_4_multi_seed.py`: `speech_commands_dim_50000` dataset.
- `figure_4_multi_seed.py`, `figure_5.py`: `binary_speech_commands_4096` dataset.
- `figure_8_multi_seed.py`: `cifar10_binary` dataset.

### Plot results
After running the experiments, you can plot the results using the provided notebook`experiments_reproduction/plot_results.ipynb`.




## Installation Instructions
Tested with python 3.9.7.
- Install PyTorch from the [official website](https://pytorch.org/) (tested with version 1.13.0) including `torchvision`.
- The requirements.txt file includes additional requirements, which can be installed via:
- `pip install -r requirements.txt`
- Install NetworkX-METIS from the official [GitHub repository](https://github.com/networkx/networkx-metis).
- Follow the official set-up for the S4 model from the [official repository](https://github.com/HazyResearch/state-spaces), and make sure that the `state_spaces` folder is saved under the `models` folder.
- For reproducing the experiment involving graph sparsification - install `Julia` from the [official website](https://julialang.org/downloads/).
 

## Citation

For citing the paper you can use:

```
@inproceedings{alexander2023what,
  title={What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement},
  author={Alexander, Yotam and De La Vega, Nimrod and Razin, Noam and Cohen, Nadav},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
