#!/bin/bash

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

RESULTS_DIR=$DIR/results

NUM_NODES=8
NODE_GPUS=8
POLICY="consolidated-backfill"
PLACEMENT_POLICY="spread"  # Default for Kubernetes (LeastAllocated).

TRACE_A_PATH=$1
TRACE_B_PATH=$2

rm -rf $RESULTS_DIR
mkdir -p $RESULTS_DIR

source $DIR/.venv/bin/activate

function run_simulator {
    trace_path=$1

    trace_filename=$(basename -- "$trace_path")
    trace_name="${trace_filename%.*}"

    result_prefix="$RESULTS_DIR/$trace_name"

    echo "Running $trace_name"

    uv run kavier cluster --jobs $trace_path --policy $POLICY  --placement $PLACEMENT_POLICY --oversized strict --num-nodes $NUM_NODES --node-gpus $NODE_GPUS --out "$result_prefix"_per_jobs.csv --out-nodes "$result_prefix"_per_nodes.csv --plot "$result_prefix"_timeline.pdf > "$result_prefix"_per_cluster.json

    echo ""
}

run_simulator $TRACE_A_PATH
run_simulator $TRACE_B_PATH
