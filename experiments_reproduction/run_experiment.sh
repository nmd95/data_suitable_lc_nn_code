#!/bin/bash

run_set() {
    devices=($1)
    seeds=($2)
    script_name=$3
    # create an array of arguments with the device and seed
    args=()
    for ((i=0;i<${#devices[@]};++i)); do
        args+=("--device ${devices[i]} --seed ${seeds[i]}")
    done

    # run the python script with the arguments in parallel
    for arg in "${args[@]}"
    do
        python $script_name $arg &
    done

    wait
}

# Check for required arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <device_numbers> <seeds> <script_id>"
    exit 1
fi

device_numbers=$1
seeds=$2
script_id=$3

# Set the job script based on the script ID
job_script="/experiments_reproduction/experiments_protocols/${script_id}_multi_seed.py"

run_set "$device_numbers" "$seeds" "$job_script"


# Example usage:
# ./modified_script.sh "1 2 3 4 5" "0 1 2 3 4" "table_2_dna"