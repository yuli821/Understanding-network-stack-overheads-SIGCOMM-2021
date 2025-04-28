#!/bin/bash
# Get the dir of this project
DIR=$(realpath $(dirname $(readlink -f $0)))

# Parse arguments
# Example: ./one-to-one.sh 128.84.155.115 192.168.10.115 enp37s0f1
public_dst_ip=${1:-192.17.100.155}
device_dst_ip=${2:-192.168.200.20}
iface=${3:-ens1f0np0}
results_dir=${4:-$DIR/results}

# Create results directory
mkdir -p $results_dir
for i in 32; do
        $DIR/multi_stream_test.py --interface iface --sender --addr $device_dst_ip --receiver_addr $public_dst_ip --config one-to-one --bind_app --cpus $i --throughput --utilisation --cache-miss --latency --output $results_dir/one-to-one_${i} | tee $results_dir/one-to-one_${i}.log
done