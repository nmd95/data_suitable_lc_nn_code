#!/usr/bin/env julia

# Load necessary packages
import Pkg

packages = [
    "NPZ",
    "Laplacians",
    "CSV",
    "DataFrames",
    "SimpleWeightedGraphs",
    "LightGraphs",
    "Graphs",
    "ArgParse"
]

for package in packages
    Pkg.add(package)
end

using NPZ
using Laplacians
using SparseArrays
using Random
using LinearAlgebra
using Statistics
using DelimitedFiles
using CSV
using DataFrames
using LightGraphs
using Graphs, SimpleWeightedGraphs
using ArgParse

function main(args)
    s = ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "load_path"
            arg_type = String
            help = "Path to the input file (corr_matrix.npy)"
        "save_dir"
            arg_type = String
            help = "Path to the directory where the output file will be saved"
    end

    parsed_args = parse_args(args, s)
    load_path = parsed_args["load_path"]
    save_dir = parsed_args["save_dir"]

    # Load the data and convert to sparse matrix
    pm = npzread(load_path)
    pm[diagind(pm)] .= 0
    pm = sparse(pm)

    # Define parameters
    EPSILON = 0.15

    # Save the sparsified matrix
    save_path = save_dir * "/corr_matrix_sparsified_epsilon_" * string(EPSILON) * ".npz"
    Gsparse = sparsify(pm, ep=EPSILON)
    Gdense = Matrix(Gsparse)
    # print shape of Gdense
    println(size(Gdense))
    npzwrite(save_path, Gdense)
end

isinteractive() || main(ARGS)
