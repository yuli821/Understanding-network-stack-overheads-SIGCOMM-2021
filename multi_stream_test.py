import subprocess as _sp
import shlex
import os
import threading
import xmlrpc.server
import argparse
import time
import signal
from process_output import *

# System Constants

# In the default IRQ affinity config mode
# ID of the RX queue for each CPU 0-31
# CPU_TO_RX_QUEUE_MAP = [0, 6, 7, 8, 1, 9, 10, 11, 2, 12, 13, 14, 3, 15, 16, 17, 4, 18, 19, 20, 5, 21, 22, 23, 31, 30, 29, 28, 27, 26, 25, 24]
# AFFINITY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0]

# DDIO IO WAYS LLC mm register location
DDIO_REG = 0xc8b

# Port for running the coordination service
COMM_PORT = 50000

# Base port of using iperf and netperf
BASE_PORT = 30000
ADDITIONAL_BASE_PORT = 40000

# Maximum number of ntuple filters
MAX_RULE_LOC = 1023

# Maximum number of CPUs/connections
# NOTE: This should be the same as number of CPUs
MAX_CPUS = 32
CPUS = list(range(MAX_CPUS))
MAX_CONNECTIONS = MAX_CPUS
MAX_RPCS = 32

# Path to executables of profiling tools
PERF_PATH = "/usr/bin/perf"
# FLAME_PATH = "/opt/FlameGraph"

# For debugging
class subprocess:
    PIPE = _sp.PIPE
    DEVNULL = _sp.DEVNULL
    STDOUT = _sp.STDOUT
    VERBOSE = False

    @staticmethod
    def enable_logging():
        subprocess.VERBOSE = True

    @staticmethod
    def Popen(*args, **kwargs):
        if subprocess.VERBOSE: print("+ " + " ".join(shlex.quote(s) for s in args[0]))
        return _sp.Popen(*args, **kwargs)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run TCP measurement experiments on the receiver.")
    # Add arguments
    #setup parameters
    parser.add_argument('interface', type=str, help='The network device interface to configure.')
    parser.add_argument('--sender', action='store_true', default=None, help='This is the sender.')
    parser.add_argument('--receiver', action='store_true', default=None, help='This is the receiver.')
    parser.add_argument("--receiver_addr", type=str, help="Address of the receiver to communicate metadata.")
    parser.add_argument("--addr", type=str, help="Address of the receiver to communicate metadata.")
    parser.add_argument("--bind_app", action='store_true', default=None, help="Whether to bind receiving thread to the specific core")
    parser.add_argument("--bind_queue", action='store_true', default=None, help="Whether to bind TX/RX queue/IRQ to the specific core")
    parser.add_argument("--config", choices=["one-to-one", "single"], default="single", help="Configuration to run the experiment with.")
    parser.add_argument("--cpus", type=int, nargs="*", help="Which CPUs to use for experiment.")
    parser.add_argument("--num-rpcs", type=int, default=0, help="Number of short flows (for mixed flow type).")
    parser.add_argument("--rpc-size", type=int, default=4000, help="Size of the RPC for short flows.")
    #run parameter
    parser.add_argument("--output", type=str, default=None, help="Write raw output to the directory.")
    parser.add_argument("--throughput", action="store_true", help="Measure throughput.")
    parser.add_argument("--utilisation", action="store_true", help="Measure CPU utilisation.")
    parser.add_argument("--cache-miss", action="store_true", help="Measure LLC miss rate.")
    parser.add_argument("--latency", action="store_true", help="Calculate the average data copy latency for each packet.")

    # Parse and verify arguments
    args = parser.parse_args()

    # Set IRQ processing CPUs
    if args.config in ["single"]:
        args.cpulist = [1]
    elif args.config in ["one-to-one"]:
        args.cpulist = [cpu for cpu in args.cpus]
    else:
        args.cpulist = []

    # Create the directory for writing raw outputs
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    # Return parsed and verified arguments
    return args

# Convenience functions
def on_or_off(state):
    return "on" if state else "off"

def stop_irq_balance():
    os.system("service irqbalance stop")

def start_irq_balance():
    os.system("service irqbalance start")

def manage_ntuple(iface, enabled):
    os.system("ethtool -K {} ntuple {}".format(iface, on_or_off(enabled)))

def ntuple_send_port_to_queue(iface, port, n, loc):
    os.system("ethtool -U {} flow-type tcp4 dst-port {} action {} loc {}".format(iface, port, n, loc))
    os.system("ethtool -U {} flow-type tcp4 src-port {} action {} loc {}".format(iface, port, n, MAX_RULE_LOC - loc))

def ntuple_send_all_traffic_to_queue(iface, n, loc):
    os.system("ethtool -U {} flow-type tcp4 action {} loc {}".format(iface, n, loc))

def manage_rps(iface, enabled):
    num_rps = 32768 if enabled else 0
    os.system("echo {} > /proc/sys/net/core/rps_sock_flow_entries".format(num_rps))
    os.system("for f in /sys/class/net/{}/queues/rx-*/rps_flow_cnt; do echo {} > $f; done".format(iface, num_rps))

def ntuple_clear_rules(iface):
    for i in range(MAX_RULE_LOC + 1):
        os.system("ethtool -U {} delete {} 2> /dev/null > /dev/null".format(iface, i))

def set_irq_affinity(iface):
    os.system("set_irq_affinity.sh {} 2> /dev/null > /dev/null".format(iface))

def setup_irq_mode_no_arfs_sender(cpulist, iface, config, bind_queue):
    stop_irq_balance() # stop dynamic core to rx queue mapping
    manage_rps(iface, False) # 1. disable the receive packet steering, so the interrupt core will be the same core that runs kernel stack processing
    ntuple_clear_rules(iface)
    set_irq_affinity(iface) # 2. core to rx queue mapping, one-to-one (in our case)

    # For single flow or outcast, we have to send all traffic to core 1;
    # for one-to-one and outcast, we use flow steering to the next core;
    # otherwise we just use RSS
    if config in ["single"]:
        manage_ntuple(iface, True)
        ntuple_send_all_traffic_to_queue(iface, 1, 0) # map all traffic to queue 1
    elif config in ["one-to-one"]:
        manage_ntuple(iface, True)
        for index, cpu in enumerate(cpulist): # one queue per CPU
            ntuple_send_port_to_queue(iface, BASE_PORT + index, cpu, index) # 3. mapping the flow to a specific queue, flow 0 -> queue 0, queue 0 -> core 0 by 1 and 2
    else:
        manage_ntuple(iface, False) # Depends on hardware RSS


def setup_irq_mode_no_arfs_receiver(cpulist, iface, config, bind_queue):
    stop_irq_balance()
    manage_rps(iface, False)
    ntuple_clear_rules(iface)
    set_irq_affinity(iface)

    # For single flow or incast, we have to send all traffic to core 1;
    # for one-to-one and outcast, we use flow steering to next core;
    # otherwise we just use RSS
    if config in ["single"]:
        manage_ntuple(iface, True)
        ntuple_send_all_traffic_to_queue(iface, 1, 0)
    elif config in ["one-to-one"]:
        manage_ntuple(iface, True)
        for index, cpu in enumerate(cpulist):
            ntuple_send_port_to_queue(iface, BASE_PORT + index, cpu, index)
    else:
        manage_ntuple(iface, False)

def setup_affinity_mode(iface, cpulist, sender, receiver, bind_app, bind_queue, config):
        if sender is not None and sender:
            setup_irq_mode_no_arfs_sender(cpulist, iface, config, bind_app, bind_queue)
        elif receiver is not None and receiver:
            setup_irq_mode_no_arfs_receiver(cpulist, iface, config, bind_app, bind_queue)
    
def run_iperf_recv(cpu, port, window, bind_app):
    if bind_app:
        if window is None:
            args = ["taskset", "-c", str(cpu), "iperf", "-i", "1", "-s", "-p", str(port)]
        else:
            args = ["taskset", "-c", str(cpu), "iperf", "-s", "-i", "1", "-p", str(port), "-w", str(window / 2) + "K"]
    else:
        if window is None:
            args = ["iperf", "-i", "1", "-s", "-p", str(port)]
        else:
            args = ["iperf", "-s", "-i", "1", "-p", str(port), "-w", str(window / 2) + "K"]

    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)


def run_netperf_recv(cpu, port):
    args = ["taskset", "-c", str(cpu), "netserver", "-p", str(port), "-D", "f"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)

# We run one iperf server process per flow, and one netserver process per CPU
def run_flows_recv(config, cpus, window, bind_app):
    procs = []
    if config == "single":
        procs.append(run_iperf_recv(cpus[0], BASE_PORT, window, bind_app))
    elif config in "one-to-one":
        procs += [run_iperf_recv(cpu, BASE_PORT + n, window, bind_app) for n, cpu in enumerate(cpus)]
    return procs

def run_iperf_send(cpu, addr, port, duration, window):
    if window is None:
        args = ["taskset", "-c", str(cpu), "iperf", "-i", "1", "-c", addr, "-t", str(duration), "-p", str(port)]
    else:
        args = ["taskset", "-c", str(cpu), "iperf", "-i", "1", "-c", addr, "-t", str(duration), "-p", str(port), "-w", str(window / 2) + "K"]

    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)


def run_netperf_send(cpu, addr, port, duration, rpc_size):
    args = ["taskset", "-c", str(cpu), "netperf", "-H", addr, "-t", "TCP_RR", "-l", str(duration), "-p", str(port), "-f", "g", "--", "-r", "{0},{0}".format(rpc_size), "-o", "throughput"]

    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)


# We run one iperf client process per flow, and one netperf process per flow
def run_flows_send(flow_type, config, addr, num_connections, num_rpcs, cpus, duration, window, rpc_size):
    procs = []
    if config == "single":
        procs.append(run_iperf_send(cpus[0], addr, BASE_PORT, duration, window))
    elif config in "one-to-one":
        procs += [run_iperf_send(cpu, addr, BASE_PORT + n, duration, window) for n, cpu in enumerate(cpus)]

    return procs

def run_perf_cache(cpus):
    args = [PERF_PATH, "stat", "-C", ",".join(map(str, set(cpus))), "-e", "LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)


def latency_measurement(enabled):
    os.system("echo {} > /sys/module/tcp/parameters/measure_latency_on".format(int(enabled)))

def run_sar(cpus):
    args = ["sar", "-u", "-P", ",".join(map(str, set(cpus))), "1", "1000"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)

def run_perf_report(perf_data_file):
    args = ["bash", "-c", "{} report --stdio --stdio-color never --percent-limit 0.01 -i {} | cat".format(PERF_PATH, perf_data_file)]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)

def dmesg_clear():
    os.system("dmesg -c > /dev/null 2> /dev/null")

def run_dmesg(level="info"):
    args = ["dmesg", "-c", "-l", level]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, universal_newlines=True)

# Convenience functions
def clear_processes():
    os.system("pkill iperf")
    os.system("pkill netserver")
    os.system("pkill netperf")
    os.system("pkill perf")
    os.system("pkill sar")

# Functions to query/set synchronization events
def mark_sender_ready():
    __sender_ready.set()
    return True


def is_receiver_ready():
    __receiver_ready.wait()
    __receiver_ready.clear()
    return True


def mark_receiver_ready():
    __receiver_ready.set()
    return True


def is_sender_ready():
    __sender_ready.wait()
    __sender_ready.clear()
    return True


def mark_sender_done():
    __sender_done.set()
    return True


def is_sender_done():
    __sender_done.wait()
    __sender_done.clear()
    return True


def get_results():
    return __results

def network_setup(iface, cpulist, server, receiver, bind_app, bind_queue, config):
    setup_affinity_mode(iface, cpulist, server, receiver, bind_app, bind_queue, config)

args = parse_args()
# network setup
network_setup(args.interface, args.cpulist, args.server, args.receiver, args.bind_app, args.bind_queue, args.config)
time.sleep(2) #sleep for 2 secs to make sure everything setup
if args.receiver:
    # Need to synchronize with the sender before starting experiment
    server = xmlrpc.server.SimpleXMLRPCServer(("0.0.0.0", COMM_PORT), logRequests=False)
    server.register_introspection_functions()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)

    # Event objects to synchronize sender and receiver
    __sender_ready = threading.Event()
    __receiver_ready = threading.Event()
    __sender_done = threading.Event()


    # Stores the results from the receiver
    __results = {}

    server_thread.start()

    # Print the output directory
    if args.output is not None:
        print("[output] writing results to {}".format(args.output))

    clear_processes()
    header = []
    output = []
    if args.throughput:
        # Wait till sender starts
        is_sender_ready()
        print("[throughput] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_recv( args.config, args.cpuslist, args.window, args.bind_app)

        # Wait till sender is done sending
        mark_receiver_ready()
        is_sender_done()

        # Kill all the processes
        for p in procs:
            p.kill()
        print("[throughput] finished experiment.")

        # Process and write the raw output
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "throughput_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)

    if args.utilisation:
        # Wait till sender starts
        is_sender_ready()
        print("[utilisation] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_recv( args.config, args.cpuslist, args.window, args.bind_app)

        # Start the sar instance
        sar = run_sar(list(set(args.cpuslist)))

        # Wait till sender is done sending
        mark_receiver_ready()
        is_sender_done()

        # Kill sar
        sar.send_signal(signal.SIGINT)
        sar.wait()

        # Kill all the processes
        for p in procs:
            p.kill()
        print("[utilisation] finished experiment.")

        # Process and write the raw output
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "utilisation_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)

        lines = sar.stdout.readlines()
        cpu_util = sum(process_util_output(lines).values())
        __results["cpu_util"] = cpu_util
        if args.output is not None:
            with open(os.path.join(args.output, "utilisation_sar.log"), "w") as f:
                f.writelines(lines)

        # Print the output
        print("[utilisation] utilisation: {:.3f}".format(cpu_util))
        header.append("receiver utilisation (%)")
        output.append("{:.3f}".format(cpu_util))

    if args.cache_miss:
        # Wait till sender starts
        is_sender_ready()
        print("[cache miss] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_recv(args.flow_type, args.config, args.num_connections, args.cpuslist, args.window)

        # Start the perf instance
        perf = run_perf_cache(list(set(args.cpuslist)))

        # Wait till sender is done sending
        mark_receiver_ready()
        is_sender_done()

        # Kill perf
        perf.send_signal(signal.SIGINT)
        perf.wait()

        # Kill all the processes
        for p in procs:
            p.kill()
        print("[cache miss] finished experiment.")

        # Process and write the raw output
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "cache-miss_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)

        lines = perf.stdout.readlines()
        cache_miss = process_cache_miss_output(lines)
        __results["cache_miss"] = cache_miss
        if args.output is not None:
            with open(os.path.join(args.output, "cache-miss_perf.log"), "w") as f:
                f.writelines(lines)

        # Print the output
        print("[cache miss] cache miss: {:.3f}".format(cache_miss))
        header.append("receiver cache miss (%)")
        output.append("{:.3f}".format(cache_miss))

    if args.latency:
        # Clear dmesg
        dmesg_clear()

        # Enable latency measurement
        latency_measurement(enabled=True)

        # Wait till sender starts
        is_sender_ready()
        print("[latency] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_recv(args.flow_type, args.config, args.num_connections, args.cpuslist, args.window)

        # Wait till sender is done sending
        mark_receiver_ready()
        is_sender_done()

        # Kill all the processes
        for p in procs:
            p.kill()
        print("[latency] finished experiment.")

        # Disable latency measurement
        latency_measurement(enabled=False)

        # Process and write the raw output
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "latency_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)

        # Start a dmesg instance to read the kernel logs
        dmesg = run_dmesg()
        lines = []
        while True:
            new_lines = dmesg.stdout.readlines()
            lines += new_lines
            if len(new_lines) == 0 and dmesg.poll() != None:
                break
        avg_latency, tail_latency = process_latency_output(lines)
        __results["avg_latency"] = avg_latency
        __results["tail_latency"] = tail_latency
        if args.output is not None:
            with open(os.path.join(args.output, "latency_dmesg.log"), "w") as f:
                f.writelines(lines)

        # Print the output
        print("[latency] avg. data copy latency: {:.3f}\ttail data copy latency: {}".format(avg_latency, tail_latency))
        header.append("avg. data copy latency (us)")
        output.append("{:.3f}".format(avg_latency))
        header.append("tail data copy latency (us)")
        output.append("{}".format(tail_latency))
elif args.sender:
    # Create the XMLRPC proxy
    receiver = xmlrpc.client.ServerProxy("http://{}:{}".format(args.receiver_addr, COMM_PORT), allow_none=True)
    # Wait till receiver is ready
    while True:
        try:
            receiver.system.listMethods()
            break
        except ConnectionRefusedError:
            time.sleep(1)
    # Print the output directory
    if args.output is not None:
        print("[output] writing results to {}".format(args.output))
    # Run the experiments
    clear_processes()
    header = []
    output = []
    if args.throughput:
        # Wait till receiver starts
        receiver.mark_sender_ready()
        receiver.is_receiver_ready()
        print("[throughput] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_send(args.flow_type, args.config, args.addr, args.num_connections, args.num_rpcs, args.cpuslist, args.duration, args.window, args.rpc_size)

        # Wait till all experiments finish
        for p in procs:
            p.wait()

        # Sender is done sending
        receiver.mark_sender_done()
        print("[throughput] finished experiment.")

        # Process and write the raw output
        total_throughput = 0
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "throughput_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)
            total_throughput += process_throughput_output(lines)

        # Print the output
        print("[throughput] total throughput: {:.3f}".format(total_throughput))
        header.append("throughput (Gbps)")
        output.append("{:.3f}".format(total_throughput))

    if args.utilisation:
        # Wait till receiver starts
        receiver.mark_sender_ready()
        receiver.is_receiver_ready()
        print("[utilisation] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_send(args.flow_type, args.config, args.addr, args.num_connections, args.num_rpcs, args.cpuslist, args.duration, args.window, args.rpc_size)

        # Start the sar instance
        sar = run_sar(list(set(args.cpuslist)))

        # Wait till all experiments finish
        for p in procs:
            p.wait()

        # Sender is done sending
        receiver.mark_sender_done()

        # Kill the sar instance
        sar.send_signal(signal.SIGINT)
        sar.wait()
        print("[utilisation] finished experiment.")

        # Process and write the raw output
        throughput = 0
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "utilisation_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)
            throughput += process_throughput_output(lines)

        lines = sar.stdout.readlines()
        cpu_util = sum(process_util_output(lines).values())
        if args.output is not None:
            with open(os.path.join(args.output, "utilisation_sar.log"), "w") as f:
                f.writelines(lines)

        # Print the output
        print("[utilisation] total throughput: {:.3f}\tutilisation: {:.3f}".format(throughput, cpu_util))
        header.append("sender utilisation (%)")
        output.append("{:.3f}".format(cpu_util))

    if args.cache_miss:
        # Wait till receiver starts
        receiver.mark_sender_ready()
        receiver.is_receiver_ready()
        print("[cache miss] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_send(args.flow_type, args.config, args.addr, args.num_connections, args.num_rpcs, args.cpuslist, args.duration, args.window, args.rpc_size)

        # Start the perf instance
        perf = run_perf_cache(list(set(args.cpuslist)))

        # Wait till all experiments finish
        for p in procs:
            p.wait()

        # Sender is done sending
        receiver.mark_sender_done()

        # Kill the perf instance
        perf.send_signal(signal.SIGINT)
        perf.wait()
        print("[cache miss] finished experiment.")

        # Process and write the raw output
        throughput = 0
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "cache-miss_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)
            throughput += process_throughput_output(lines)

        lines = perf.stdout.readlines()
        cache_miss = process_cache_miss_output(lines)
        if args.output is not None:
            with open(os.path.join(args.output, "cache-miss_perf.log"), "w") as f:
                f.writelines(lines)

        # Print the output
        print("[cache miss] total throughput: {:.3f}\tcache miss: {:.3f}".format(throughput, cache_miss))
        header.append("sender cache miss (%)")
        output.append("{:.3f}".format(cache_miss))
    if args.latency:
        # Wait till receiver starts
        receiver.mark_sender_ready()
        receiver.is_receiver_ready()
        print("[latency] starting experiment...")

        # Start iperf and/or netperf instances
        procs = run_flows_send(args.flow_type, args.config, args.addr, args.num_connections, args.num_rpcs, args.cpuslist, args.duration, args.window, args.rpc_size)

        # Wait till all experiments finish
        for p in procs:
            p.wait()

        # Sender is done sending
        receiver.mark_sender_done()
        print("[latency] finished experiment.")

        # Process and write the raw output
        throughput = 0
        for i, p in enumerate(procs):
            lines = p.stdout.readlines()
            if args.output is not None:
                with open(os.path.join(args.output, "latency_benchmark_{}.log".format(i)), "w") as f:
                    f.writelines(lines)
            throughput += process_throughput_output(lines)

        # Print the output
        print("[latency] total throughput: {:.3f}".format(throughput))
