# TODO
from pathlib import Path
import sys
import os
import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the current directory (where Python is called) to sys.path. This is
# required to load modules in the project root directory (like dfaas_utils.py).
sys.path.append(os.getcwd())

import dfaas_env
import plot_utils  # To remove annoying debug messages from matplotlib.


def reward_asym_no_forward_total_reject(out_dir):
    queue_cap_max = 100
    local_reqs = 20

    x = np.arange(20, queue_cap_max)

    rewards = np.zeros(x.size)
    idx = 0
    for reject_reqs in x:
        if local_reqs <= queue_cap_max:
            reward = dfaas_env.DFaaS_ASYM._calculate_reward_1_v2((local_reqs, reject_reqs), (0,), (local_reqs, queue_cap_max))
        else:
            local_excess = local_reqs - queue_cap_max
            reward = dfaas_env.DFaaS_ASYM._calculate_reward_1_v2((local_reqs, reject_reqs), (local_excess,), (queue_cap_max, queue_cap_max))

        rewards[idx] = reward
        idx += 1

    fig = plt.figure(dpi=600, layout="constrained")
    ax = fig.subplots()

    ax.plot(x, rewards)

    ax.set_xlabel("Richieste rifiutate")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_ylabel("Ricompensa")
    ax.yaxis.set_ticks(np.arange(0, 1+.1, .1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))

    ax.grid(axis="both")
    ax.set_axisbelow(True)

    # Save the plot.
    path = out_dir / "reward_asym_no_forward_total_reject.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def reward_asym_no_forward_local_reqs(out_dir):
    queue_cap_max = 100
    local_max = 150

    x = np.arange(0, local_max)

    rewards = np.empty(local_max)
    for local_reqs in x:
        if local_reqs <= queue_cap_max:
            reward = dfaas_env.DFaaS_ASYM._calculate_reward_1_v2((local_reqs, 0), (0,), (local_reqs, queue_cap_max))
        else:
            local_excess = local_reqs - queue_cap_max
            reward = dfaas_env.DFaaS_ASYM._calculate_reward_1_v2((local_reqs, 0), (local_excess,), (queue_cap_max, queue_cap_max))

        rewards[local_reqs] = reward

    fig = plt.figure(dpi=600, layout="constrained")
    ax = fig.subplots()

    ax.plot(x, rewards)

    ax.set_xlabel("Richieste locali")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_ylabel("Ricompensa")
    ax.yaxis.set_ticks(np.arange(0, 1, .1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))

    ax.grid(axis="both")
    ax.set_axisbelow(True)

    # Save the plot.
    path = out_dir / "reward_asym_no_forward_local_reqs.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def reward_fw_forward_reject(out_dir):
    queue_cap_max = 100

    reject_reqs = np.arange(0, 150)
    forward_rejects = [0, 10, 30, 50]
    local_reqs = 100
    forward_reqs = 50

    rewards = {forward: np.empty(reject_reqs.size) for forward in forward_rejects}
    for forward_reject in forward_rejects:
        for rejects in reject_reqs:
            reward = dfaas_env.DFaaS._calculate_reward_v2((local_reqs, forward_reqs, rejects),
                                                          (0, forward_reject),
                                                          (local_reqs, queue_cap_max))

            rewards[forward_reject][rejects] = reward

    fig = plt.figure(dpi=600, layout="constrained")
    ax = fig.subplots()

    for forward_reject in rewards:
        ax.plot(reject_reqs, rewards[forward_reject], label=f"forward_reject = {forward_reject}")

    ax.set_xlabel("Richieste rifiutate")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_ylabel("Ricompensa")
    ax.yaxis.set_ticks(np.arange(0, 1, .1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))

    ax.legend()

    ax.grid(axis="both")
    ax.set_axisbelow(True)

    # Save the plot.
    path = out_dir / "reward_fw_forward_reject.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def reward_fw_can_forward(out_dir):
    queue_cap_max = 100

    reject_reqs = np.arange(0, 150)
    forward_reject = 0
    local_reqs = [0, 40, 90, 110]
    forward_reqs = 0

    rewards = {local: np.empty(reject_reqs.size) for local in local_reqs}
    for local in local_reqs:
        for rejects in reject_reqs:
            if local <= queue_cap_max:
                reward = dfaas_env.DFaaS._calculate_reward_v2((local, forward_reqs, rejects),
                                                              (0, forward_reject),
                                                              (local, queue_cap_max))
            else:
                local_excess = local - queue_cap_max
                reward = dfaas_env.DFaaS._calculate_reward_v2((local, forward_reqs, rejects),
                                                              (local_excess, forward_reject),
                                                              (queue_cap_max, queue_cap_max))

            rewards[local][rejects] = reward

    fig = plt.figure(dpi=600, layout="constrained")
    ax = fig.subplots()

    for local in rewards:
        ax.plot(reject_reqs, rewards[local], label=f"local_reqs = {local}")

    ax.set_xlabel("Richieste rifiutate")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_ylabel("Ricompensa")
    ax.yaxis.set_ticks(np.arange(0, 1, .1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))

    ax.legend()

    ax.grid(axis="both")
    ax.set_axisbelow(True)

    # Save the plot.
    path = out_dir / "reward_fw_can_forward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def reward_fw_cant_forward(out_dir):
    queue_cap_max = 100

    reject_reqs = np.arange(0, 150)
    forward_reject = 1
    local_reqs = [0, 40, 90, 110]
    forward_reqs = 1

    rewards = {local: np.empty(reject_reqs.size) for local in local_reqs}
    for local in local_reqs:
        for rejects in reject_reqs:
            if local <= queue_cap_max:
                reward = dfaas_env.DFaaS._calculate_reward_v2((local, forward_reqs, rejects),
                                                              (0, forward_reject),
                                                              (local, queue_cap_max))
            else:
                local_excess = local - queue_cap_max
                reward = dfaas_env.DFaaS._calculate_reward_v2((local, forward_reqs, rejects),
                                                              (local_excess, forward_reject),
                                                              (queue_cap_max, queue_cap_max))

            rewards[local][rejects] = reward

    fig = plt.figure(dpi=600, layout="constrained")
    ax = fig.subplots()

    for local in rewards:
        ax.plot(reject_reqs, rewards[local], label=f"local_reqs = {local}")

    ax.set_xlabel("Richieste rifiutate")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.set_ylabel("Ricompensa")
    ax.yaxis.set_ticks(np.arange(0, 1, .1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))

    ax.legend()

    ax.grid(axis="both")
    ax.set_axisbelow(True)

    # Save the plot.
    path = out_dir / "reward_fw_cant_forward.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


def make(out_dir):
    out_dir.mkdir(exist_ok=True)

    reward_asym_no_forward_local_reqs(out_dir)

    reward_asym_no_forward_total_reject(out_dir)

    reward_fw_forward_reject(out_dir)

    reward_fw_can_forward(out_dir)

    reward_fw_cant_forward(out_dir)

if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Create parser and parse arguments.
    parser = argparse.ArgumentParser(prog="dfaas_env_reward")

    parser.add_argument(dest="output_directory",
                        help="Output directory where to save the plots",
                        type=Path)

    args = parser.parse_args()

    make(args.output_directory.resolve())
