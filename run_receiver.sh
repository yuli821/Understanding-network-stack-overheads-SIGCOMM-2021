#!/bin/bash
# Get the dir of this project
DIR=$(realpath $(dirname $(readlink -f $0)))

# Parse arguments
# Example: ./one-to-one.sh enp37s0f1
iface=${1:-ens3f1np1}
results_dir=${2:-$DIR/results}

for i in 28; do
        # $DIR/multi_stream_test.py $iface --config one-to-one --receiver --bind_app --cpus $i --throughput --utilisation --cache-miss --output $results_dir/one-to-one_${i} | tee $results_dir/one-to-one_${i}.log
        $DIR/multi_stream_test.py $iface --config one-to-one --receiver --cpus $i --throughput --utilisation --cache-miss --output $results_dir/one-to-one_${i} | tee $results_dir/one-to-one_${i}.log
done
