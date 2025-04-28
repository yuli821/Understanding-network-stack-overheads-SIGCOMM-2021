#!/bin/bash
# Get the dir of this project
DIR=$(realpath $(dirname $(readlink -f $0))/../..)

# Parse arguments
# Example: ./one-to-one.sh enp37s0f1
iface=${1:-ens3f1np1}
results_dir=${2:-$DIR/results}

for i in 32; do
        $DIR/multi_stream_test.py --config one-to-one --receiver --bind_app --cpus $i --throughput --utilisation --cache-miss --latency --output $results_dir/one-to-one_${i} | tee $results_dir/one-to-one_${i}.log
done
