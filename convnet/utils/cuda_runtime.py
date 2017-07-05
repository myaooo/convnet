import os
import re
import subprocess

import tensorflow as tf


_gpu_memory = 1.0


def init_tf_environ(gpu_num=0, gpu_memory_fraction=1.0):
    """
    Init CUDA environments, which the number of gpu to use
    :param gpu_num:
    :return:
    """
    global _gpu_memory
    _gpu_memory = gpu_memory_fraction
    cuda_devices = ""
    if gpu_num == 0:
        print("Not using any gpu devices.")
    else:
        try:
            best_gpus = pick_gpu_lowest_memory(gpu_num)
            cuda_devices = ",".join([str(e) for e in best_gpus])
            print("Using gpu device: {:s}".format(cuda_devices))
        except:
            cuda_devices = ""
            print("Cannot find gpu devices!")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldn't parse "+line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory(num=1):
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memorys, best_gpus = list(zip(*sorted(memory_gpu_map)[:num]))
    return best_gpus
