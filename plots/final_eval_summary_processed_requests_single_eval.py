# This Python script generates a graph showing the metrics related to the reward
# of the the evaluation phase.
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


def _get_data(exp):
    data = {"values": [], "min": [], "max": [], "types": []}

    evals_dir = list(exp.glob("evaluation_*"))
    for eval_dir in evals_dir:
        eval_type = dfaas_utils.json_to_dict(eval_dir / "env_config.json")
        eval_type = eval_type["input_requests_type"]
        avg_eval, min_eval, max_eval = _avg_processed_requests_eval(eval_dir)

        # Because matplotlib wants the errors (min, max) as offsets, I need to
        # process the values.
        min_eval = avg_eval - min_eval
        max_eval = max_eval - avg_eval

        data["values"].append(avg_eval * 100)
        data["min"].append(min_eval * 100)
        data["max"].append(max_eval * 100)
        data["types"].append(eval_type)

    return data


def make(env_type, exps, out):
    train_type = dfaas_utils.json_to_dict(exps[0] / "env_config.json")
    train_type = train_type["input_requests_type"]
    train_type_2 = dfaas_utils.json_to_dict(exps[1] / "env_config.json")
    assert train_type == train_type_2["input_requests_type"]

    data_ppo = _get_data(exps[0])
    data_ppo_cc = _get_data(exps[1])

    fig = plt.figure(figsize=(7, 5), dpi=600, layout="constrained")
    ax = fig.subplots()

    # Set bar width and colors.
    bar_width = 0.30
    ppo_bar_color = '#99ccff'
    ppo_cc_bar_color = '#ff9999'
    max_reward_line_color = "#7a49a5"
    capsize = 10  # Length of the error horizontal bar.

    x_ticks = np.arange(len(data_ppo["values"]))
    ppo_bars = ax.bar(x_ticks - bar_width/2, data_ppo["values"], bar_width,
                      yerr=[data_ppo["min"], data_ppo["max"]],
                      capsize=capsize, color=ppo_bar_color)
    ppo_cc_bars = ax.bar(x_ticks + bar_width/2, data_ppo_cc["values"], bar_width,
                         yerr=[data_ppo_cc["min"], data_ppo_cc["max"]],
                         capsize=capsize, color=ppo_cc_bar_color)

    # Set label for both axis.
    ax.set_ylabel("Richieste processate", fontsize="large")
    ax.set_xlabel("Scenario di valutazione", fontsize="large")

    # Y axis is a percentual.
    bottom = 20  # Heuristically selected.
    ax.set_ylim(bottom=bottom, top=100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    # Show percent tick every 10%.
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Set and show legend.
    ax.legend((ppo_bars, ppo_cc_bars), ('PPO', 'PPO-CC'))

    # Set X axis labels (one for each pair of PPO and PPO-CC experiments).
    assert data_ppo["types"] == data_ppo_cc["types"]
    assert data_ppo["types"][0] == "synthetic-sinusoidal"
    assert data_ppo["types"][1] == "synthetic-normal"
    assert data_ppo["types"][2] == "real"
    labels = ["Sintetico sinusoidale",
              "Sintetico gaussiano",
              "Reale"]
    ax.set_xticks(x_ticks, labels=labels, fontsize='medium', fontstretch='condensed')

    # Set background.
    ax.grid(axis="both")
    ax.set_axisbelow(True)  # By default the axis is over the content.

    # Save the plot.
    path = out / f"eval_{env_type}_train_{train_type}_processed_requests.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path.as_posix()!r}")


if __name__ == "__main__":
    matplotlib.use("pdf", force=True)

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    sym_real = [Path(f"results/final/DFAAS-MA_SYM_500_real"),
                Path(f"results/final/DFAAS-MA_SYM_500_cc_real")]

    out = Path("results/final/plots")

    make("SYM", sym_real, out)

    sym_norm = [Path(f"results/final/DFAAS-MA_SYM_500_synt_norm"),
                Path(f"results/final/DFAAS-MA_SYM_500_cc_synt_norm")]

    make("SYM", sym_norm, out)
