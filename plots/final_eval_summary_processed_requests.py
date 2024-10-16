# This Python script generates a plot for each environment type showing the
# average total processed requests for all evaluations.
#
# Note: this script assumes that the experiment directories are located in
# "results/final".
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
import dfaas_utils
import plot_utils


def _get_exp_env_type(exp_name):
    if "BASE" in exp_name:
        return "BASE"
    if "ASYM" in exp_name:
        return "ASYM"

    return "SYM"


def _get_exp_train_scenario(exp_name):
    if "real" in exp_name:
        return "R"  # Real scenario.
    if "synt_norm" in exp_name:
        return "S"  # Sinthetic sinusoidal scenario.

    return "G"  # Sinthetic gaussian scenario.


def _processed_requests_episode(eval, env, epi_idx):
    steps = int(eval["episode_len_mean"])
    eval = eval["hist_stats"]
    input_reqs_epi = processable_reqs_epi = processed_reqs_epi = 0

    # Only environments with the same queue capacity are supported for this
    # script.
    queue_capacity = np.unique(list(env.queue_capacity_max.values()))
    assert queue_capacity.size == 1, "Environments with different queue capacity size for the agents are not supported"
    total_capacity = queue_capacity.item() * env.agents

    for step in range(steps):
        input_reqs_step = 0  # Input requests for this step.
        for agent in env.agent_ids:
            input_reqs_step += eval["observation_input_requests"][epi_idx][agent][step]

        # Processable requests for this step.
        if input_reqs_step <= total_capacity:
            processable_reqs_step = input_reqs_step
        else:
            processable_reqs_step = total_capacity

        processed_reqs_step = 0  # Processed requests for this step.
        for agent in env.agent_ids:
            local = eval["action_local"][epi_idx][agent][step]
            local_excess = eval["excess_local"][epi_idx][agent][step]
            forward = eval["action_forward"][epi_idx][agent][step]
            forward_reject = eval["excess_forward_reject"][epi_idx][agent][step]

            processed_reqs_step += (local - local_excess) + (forward - forward_reject)

        assert processed_reqs_step <= processable_reqs_step <= input_reqs_step

        input_reqs_epi += input_reqs_step
        processable_reqs_epi += processable_reqs_step
        processed_reqs_epi += processed_reqs_step

    assert processed_reqs_epi <= processable_reqs_epi <= input_reqs_epi
    return input_reqs_epi, processable_reqs_epi, processed_reqs_epi


def _avg_processed_requests_eval(eval_dir):
    # Read data from the given evaluation directory
    eval = dfaas_utils.parse_result_file(eval_dir / "evaluation.json")
    eval = eval[0]["evaluation"]
    episodes = eval["episodes_this_iter"]
    env = plot_utils.get_env(eval_dir)

    processed_reqs_ratio = []
    for epi_idx in range(episodes):
        input_reqs_epi, processable_reqs_epi, processed_reqs_epi = _processed_requests_episode(eval, env, epi_idx)

        processed_reqs_ratio.append(processed_reqs_epi / processable_reqs_epi)

    avg = np.mean(processed_reqs_ratio)
    min = np.min(processed_reqs_ratio)
    max = np.max(processed_reqs_ratio)

    return avg, min, max


def _avg_processed_requests_exp(exp_dir):
    evals_dir = list(exp_dir.glob("evaluation_*"))

    avg = min = max = 0
    for eval_dir in evals_dir:
        avg_eval, min_eval, max_eval = _avg_processed_requests_eval(eval_dir)

        avg += avg_eval
        min += min_eval
        max += max_eval

    min = min / len(evals_dir)
    max = max / len(evals_dir)
    avg = avg / len(evals_dir)

    # Because matplotlib wants the errors (min, max) as offsets, I need to
    # process the values.
    assert min <= avg <= max
    min = avg - min
    max = max - avg

    return avg, min, max


def _get_data(exps):
    data = {}

    data = {}
    for algo in ["PPO", "PPO-CC"]:
        data[algo] = {}
        data[algo]["values"] = []
        data[algo]["min"] = []
        data[algo]["max"] = []
        data[algo]["labels"] = []
        for exp_dir in exps:
            if algo == "PPO" and "cc" in exp_dir.name:
                continue
            elif algo == "PPO-CC" and "cc" not in exp_dir.name:
                continue

            scenario = _get_exp_train_scenario(exp_dir.name)

            processed_reqs_avg, processed_reqs_min_avg, processed_reqs_max_avg = _avg_processed_requests_exp(exp_dir)

            data[algo]["values"].append(processed_reqs_avg)
            data[algo]["min"].append(processed_reqs_min_avg)
            data[algo]["max"].append(processed_reqs_max_avg)
            data[algo]["labels"].append(scenario)

        # Convert lists to numpy arrays.
        data[algo]["values"] = np.asarray(data[algo]["values"])
        data[algo]["min"] = np.asarray(data[algo]["min"])
        data[algo]["max"] = np.asarray(data[algo]["max"])
        data[algo]["labels"] = np.asarray(data[algo]["labels"])

    return data


def make(env_type, exps, out):
    plots_dir = out
    plots_dir.mkdir(exist_ok=True)

    data = _get_data(exps)

    fig = plt.figure(figsize=(12, 7), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Set bar width and colors.
    bar_width = 0.30
    ppo_bar_color = '#99ccff'
    ppo_cc_bar_color = '#ff9999'
    capsize = 10  # Length of the error horizontal bar.

    x_ticks = np.arange(len(data["PPO"]["values"]))
    ppo_bars = ax.bar(x_ticks - bar_width/2, data["PPO"]["values"]*100, bar_width,
                      yerr=[data["PPO"]["min"]*100, data["PPO"]["max"]*100],
                      capsize=capsize, color=ppo_bar_color)
    ppo_cc_bars = ax.bar(x_ticks + bar_width/2, data["PPO-CC"]["values"]*100, bar_width,
                         yerr=[data["PPO-CC"]["min"]*100, data["PPO-CC"]["max"]*100],
                         capsize=capsize, color=ppo_cc_bar_color)

    # Set label for both axis.
    ax.set_ylabel("Richieste processate", fontsize="large")
    ax.set_xlabel("Scenario di addestramento", fontsize="large")

    # Y axis is a percentual.
    bottom = 20  # Heuristically selected.
    ax.set_ylim(bottom=bottom, top=100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    # Show percent tick every 10%.
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Set and show legend.
    ax.legend((ppo_bars, ppo_cc_bars), ('PPO', 'PPO-CC'))

    # Set X axis labels (one for each pair of PPO and PPO-CC experiments).
    labels = ["Reale",
              "Sintetico sinusoidale",
              "Sintetico gaussiano"]
    ax.set_xticks(x_ticks, labels=labels, fontsize="large", fontstretch='condensed')

    # Set background.
    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = plots_dir / f"eval_{env_type}_summary_processed_requests.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    # Default size is too small.
    font = {"family": "serif", "size": 20}
    matplotlib.rc('font', **font)

    epilog = """You cannot specify the path of the experiments, they are fixed
    under the "results/final" directory. The evaluations must be done before
    running this script. This script produces a plot for each type of
    scenario."""

    parser = argparse.ArgumentParser(epilog=epilog)

    args = parser.parse_args()

    exps = {}
    exps["SYM"] = [Path(f"results/final/DFAAS-MA_SYM_500_real"),
                   Path(f"results/final/DFAAS-MA_SYM_500_synt_sin"),
                   Path(f"results/final/DFAAS-MA_SYM_500_synt_norm"),
                   Path(f"results/final/DFAAS-MA_SYM_500_cc_real"),
                   Path(f"results/final/DFAAS-MA_SYM_500_cc_synt_sin"),
                   Path(f"results/final/DFAAS-MA_SYM_500_cc_synt_norm")]
    exps["ASYM"] = [Path(f"results/final/DFAAS-MA_ASYM_500_real"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_synt_sin"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_synt_norm"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_cc_real"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_cc_synt_sin"),
                    Path(f"results/final/DFAAS-MA_ASYM_500_cc_synt_norm")]
    exps["BASE"] = [Path(f"results/final/DFAAS-MA_BASE_500_real"),
                    Path(f"results/final/DFAAS-MA_BASE_500_synt_sin"),
                    Path(f"results/final/DFAAS-MA_BASE_500_synt_norm"),
                    Path(f"results/final/DFAAS-MA_BASE_500_cc_real"),
                    Path(f"results/final/DFAAS-MA_BASE_500_cc_synt_sin"),
                    Path(f"results/final/DFAAS-MA_BASE_500_cc_synt_norm")]

    out = Path("results/final/plots")

    for env_type in ["BASE", "ASYM", "SYM"]:
        make(env_type, exps[env_type], out)
