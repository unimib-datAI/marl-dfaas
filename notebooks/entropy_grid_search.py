import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import json
    import gzip

    import marimo as mo

    import numpy as np
    import pandas as pd

    import utils


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Results of entropy coeff. and decay grid search for PPO

    This notebook presents the results of a grid search aimed at finding the optimal starting value for the entropy coefficient and its decay rate in PPO.

    The experiment names and data are specific (hardcoded) and can be found under the `results/entropy` directory.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Experiments loading""")
    return


@app.cell(hide_code=True)
def _():
    exp_base_dir_widget = mo.ui.file_browser(
        initial_path=Path("results"), selection_mode="directory", multiple=False, label="Experiments base path: "
    )

    exp_base_dir_widget
    return (exp_base_dir_widget,)


@app.cell
def _(exp_base_dir_widget):
    # Wait until a directory has been selected
    mo.stop(exp_base_dir_widget.path() is None)

    # FIXME: This data loading method uses too much RAM.
    def load_experiments(base_dir):
        experiments = []
        experiments_data = {}
        evaluation_data = {}

        dirs = list(base_dir.iterdir())
        for exp_dir in mo.status.progress_bar(collection=dirs, title="Loading experiments"):
            print(exp_dir)
            # Get only the last part as experiment name.
            # As example: "ec0.005_di0.7" from "DF_20251017_103522_PPO_5_decay_ec0.005_di0.7".
            start = exp_dir.name.find("_ec")
            if start != -1:
                name = exp_dir.name[start + 1 :]  # Skip "_".
            else:
                raise ValueError("Supports only a specific experiment name format")

            # Read exp_config.json for entropy and entropy decay values.
            config_path = exp_dir / "exp_config.json"
            config = json.loads(config_path.read_text())
            entropy = config["algorithm"]["entropy_coeff"]
            entropy_decay = config["algorithm"]["entropy_coeff_decay_iterations"]

            # Read result.json.gz (JSON-L) as a list of dicts (not using pandas).
            result_path = exp_dir / "result.json.gz"
            exp_data = []
            with gzip.open(result_path, "rt") as f:
                for line in f:
                    exp_data.append(json.loads(line))
            experiments_data[name] = exp_data

            # Read evaluation.json a a list of dict. In this case it is already an array
            # of dict, one for each iteration.
            eval_path = exp_dir / "evaluation.json"
            with eval_path.open() as file:
                evaluation_data[name] = json.load(file)

            experiments.append(
                {"name": str(name), "exp_dir": str(exp_dir), "entropy": entropy, "entropy_decay": entropy_decay}
            )

        return pd.DataFrame(experiments), experiments_data, evaluation_data

    experiments_df, experiments_data, evaluation_data = load_experiments(
        exp_base_dir_widget.path().resolve().absolute()
    )
    return evaluation_data, experiments_data, experiments_df


@app.cell
def _(experiments_df):
    _entropies = experiments_df["entropy"].unique()
    _entropy_coeffs = experiments_df["entropy_decay"].unique()
    mo.md(f"""
    **Experiments**:     {experiments_df.shape[0]}  
    **Entropy coefficients**:  {_entropies}  
    **Entropy decays**:   {_entropy_coeffs}  
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Experiment selection""")
    return


@app.cell
def _(experiments_df):
    entropy_exp_filter = mo.ui.dictionary(
        label="Experiments", elements={exp_name: mo.ui.checkbox(value=True) for exp_name in experiments_df["name"]}
    )

    mo.md(f"You can filter the experiment to show in the next plot:\n{entropy_exp_filter}")
    return (entropy_exp_filter,)


@app.cell
def _(entropy_exp_filter):
    # Convert the dictionary to an array.
    filter_exp = [exp for exp in entropy_exp_filter.value if entropy_exp_filter.value[exp]]
    return (filter_exp,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Entropy metric and decay""")
    return


@app.cell
def _(experiments_data, experiments_df):
    def get_entropy_data(experiments_data):
        """
        Extracts all entropy and entropy_coeff values for each policy in each experiment iteration,
        and returns them as a pandas DataFrame with one column per policy (for entropy values),
        plus a single entropy_coeff column (if equal across all policies).
        """
        results = []

        iterations = len(experiments_data)
        for iteration_idx in range(iterations):
            data = experiments_data[iteration_idx]
            # Collect all entropy and entropy_coeff values.
            policy_entropies = {}
            entropy_coeffs = []

            for policy, policy_data in data["info"]["learner"].items():
                stats = policy_data["learner_stats"]
                policy_entropies[policy] = stats["entropy"]
                entropy_coeffs.append(stats["entropy_coeff"])

            # Check if all entropy coeff. values are the same for all policies.
            entropy_coeff_all_same = all(ec == entropy_coeffs[0] for ec in entropy_coeffs)
            if not entropy_coeff_all_same:
                raise ValueError(f"Entropy coeff. are not the same for all policies at iteration {iteration_idx}")

            # Save all entropies per policy and the single unique entropy_coeff value.
            row = {"iteration": iteration_idx, "entropy_coeff": entropy_coeffs[0]}
            row.update(policy_entropies)
            results.append(row)

        # Create dataframe with columns: iteration, entropy_coeff, <policy1>, <policy2>, ...
        df = pd.DataFrame(results)
        return df

    entropy_data = {}
    for exp in experiments_df["name"]:
        entropy_data[exp] = get_entropy_data(experiments_data[exp])
    return (entropy_data,)


@app.cell
def _(entropy_data, experiments_df):
    def make_entropy_coeff_plot(experiments_df, entropy_data):
        fig = utils.get_figure("entropy_coeff")
        ax = fig.subplots()

        for exp_name in entropy_data.keys():
            entropy_start = experiments_df[experiments_df["name"] == exp_name]["entropy"].values[0]
            entropy_decay = experiments_df[experiments_df["name"] == exp_name]["entropy"].values[0]
            label = f"E {entropy_start} D {entropy_decay}"
            ax.plot(entropy_data[exp_name]["entropy_coeff"], label=label)

        ax.set_title("Entropy coefficient decay")
        ax.set_ylabel("Entropy coefficient")
        ax.set_xlabel("Iteration")
        ax.legend()
        ax.grid(axis="both")
        ax.set_axisbelow(True)

        return mo.mpl.interactive(fig)

    make_entropy_coeff_plot(experiments_df, entropy_data)
    return


@app.cell
def _(entropy_data, filter_exp):
    def make_entropy_plot(entropy_data, filter_exp=None):
        # Get the policies by name. We assume all experiments have the same number of
        # policies.
        random_exp = next(iter(entropy_data))
        policy_names = sorted(entropy_data[random_exp].filter(regex="^policy").columns)

        # Plot only selected experiments, if filter_exp is given.
        experiments = entropy_data.keys()
        if filter_exp is not None:
            experiments = [exp for exp in filter_exp if exp in filter_exp]

        figures = []
        for policy_name in policy_names:
            fig = utils.get_figure(f"entropy_{policy_name}")
            ax = fig.subplots()

            for exp_name in experiments:
                ax.plot(entropy_data[exp_name][policy_name], label=exp_name, alpha=0.6)

            ax.set_title(f"Entropy for {policy_name}")
            ax.set_ylabel("Entropy")
            ax.set_yscale("symlog")  # We have negative values.
            ax.set_xlabel("Iteration")
            ax.legend()
            ax.grid(axis="both")
            ax.set_axisbelow(True)

            figures.append(mo.mpl.interactive(fig))

        return mo.vstack(figures)

    make_entropy_plot(entropy_data, filter_exp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Reward""")
    return


@app.function
def get_reward_data_single_exp(exp_data):
    reward_iter = []
    iterations = len(exp_data)
    for i in range(iterations):
        reward_iter.append({})

        # Get average cumulative reward per iteration. The "all" policy is the cumulative over
        # all policies.
        reward_iter[i]["all"] = exp_data[i]["env_runners"]["episode_reward_mean"]
        for policy, reward in exp_data[i]["env_runners"]["policy_reward_mean"].items():
            reward_iter[i][policy] = reward

    return pd.DataFrame(reward_iter)


@app.cell
def _(experiments_data, experiments_df):
    reward_data = {}
    for _exp in experiments_df["name"]:
        reward_data[_exp] = get_reward_data_single_exp(experiments_data[_exp])
    return (reward_data,)


@app.function
def make_reward_plot(reward_data, filter_exp=None, mode="train"):
    # Get the policies by name. We assume all experiments have the same number of
    # policies.
    random_exp = next(iter(reward_data))
    policy_names = reward_data[random_exp].columns

    # Plot only selected experiments, if filter_exp is given.
    experiments = reward_data.keys()
    if filter_exp is not None:
        experiments = [exp for exp in filter_exp if exp in filter_exp]

    figures = []
    for policy_name in policy_names:
        fig = utils.get_figure(f"reward_{mode}_{policy_name}")
        ax = fig.subplots()

        for exp_name in experiments:
            ax.plot(reward_data[exp_name][policy_name], label=exp_name, alpha=0.6)

        if policy_name == "all":
            ax.set_title(
                f"Average cumulative reward per episode for {policy_name} policies\n(cumulative over all policies)"
            )
        else:
            ax.set_title(f"Average cumulative reward per episode for {policy_name}")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Iteration")
        ax.legend()
        ax.grid(axis="both")
        ax.set_axisbelow(True)

        figures.append(mo.mpl.interactive(fig))

    return mo.vstack(figures)


@app.cell
def _(filter_exp, reward_data):
    make_reward_plot(reward_data, filter_exp, mode="train")
    return


@app.cell
def _():
    mo.md(r"""### Cumulative reward (summary)""")
    return


@app.cell
def _(reject_data):
    _random_exp = next(iter(reject_data))
    _iterations = reject_data[_random_exp].shape[0]
    reward_summary_iters_widget = mo.ui.number(
        start=1, stop=_iterations, debounce=True, label="Latest iterations to consider:"
    )
    reward_summary_iters_widget
    return (reward_summary_iters_widget,)


@app.function
def get_reward_summary(reward_data_exps, latest_iters=10):
    result = {exp_name: {} for exp_name in reward_data_exps.keys()}
    for exp_name in result:
        result[exp_name] = reward_data_exps[exp_name].tail(latest_iters).mean()

    # We transpose because we can the columns to be the policies and rows the experiments.
    return pd.DataFrame(result).T


@app.cell
def _(reward_data, reward_summary_iters_widget):
    get_reward_summary(reward_data, reward_summary_iters_widget.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Rejection Rate

    For each node, we measure the cumulative incoming rate over an episode. The rejection rate is then calculated by combining three components: the rate directly rejected by the agent, the rate rejected during local processing, and the rate rejected after being forwarded to neighboring nodes.
    """
    )
    return


@app.function
def get_reject_data_single_exp(exp_data):
    iterations = len(exp_data)

    # Get agents from a random iteration and a random episode. We assume the agents
    # keeps the same (that's true).
    agents = exp_data[0]["env_runners"]["hist_stats"]["observation_input_rate"][0].keys()

    reject_data = {agent: [] for agent in agents}
    reject_data["all"] = []

    for i in range(iterations):
        episodes = exp_data[i]["env_runners"]["episodes_this_iter"]
        assert episodes == 1, "Only one episode per iteration is supported currently"

        obs_input_rate_all, total_reject_all = 0, 0

        # Calculate percent for each agent.
        for agent in agents:
            obs_input_rate = np.sum(exp_data[i]["env_runners"]["hist_stats"]["observation_input_rate"][0][agent])

            action_reject = np.sum(exp_data[i]["env_runners"]["hist_stats"]["action_reject"][0][agent])
            local_reject = np.sum(exp_data[i]["env_runners"]["hist_stats"]["incoming_rate_local_reject"][0][agent])
            forward_reject = np.sum(exp_data[i]["env_runners"]["hist_stats"]["forward_reject_rate"][0][agent])

            total_reject = action_reject + local_reject + forward_reject

            if obs_input_rate > 0:
                reject_data[agent].append(total_reject / obs_input_rate)
            else:
                reject_data[agent].append(0)

            obs_input_rate_all += obs_input_rate
            total_reject_all += total_reject

        # Cumulative for all agents.
        if obs_input_rate_all > 0:
            reject_data["all"].append(total_reject_all / obs_input_rate_all)
        else:
            reject_data["all"].append(0)

    return pd.DataFrame(reject_data)


@app.cell
def _(experiments_data, experiments_df):
    reject_data = {}
    for _exp in experiments_df["name"]:
        reject_data[_exp] = get_reject_data_single_exp(experiments_data[_exp])
    return (reject_data,)


@app.function
def make_reject_plot(reject_data, filter_exp=None, mode="train"):
    from matplotlib.ticker import PercentFormatter

    # Get the policies by name. We assume all experiments have the same number of
    # policies.
    random_exp = next(iter(reject_data))
    policy_names = sorted(reject_data[random_exp].columns)

    # Plot only selected experiments, if filter_exp is given.
    experiments = reject_data.keys()
    if filter_exp is not None:
        experiments = [exp for exp in filter_exp if exp in filter_exp]

    figures = []
    for policy_name in policy_names:
        fig = utils.get_figure(f"reject_{mode}_{policy_name}")
        ax = fig.subplots()

        for exp_name in experiments:
            ax.plot(reject_data[exp_name][policy_name], label=exp_name, alpha=0.6)

        if policy_name == "all":
            ax.set_title(f"Cumulative reject per episode for {policy_name} policies\n(cumulative over all policies)")
        else:
            ax.set_title(f"Cumulative reject per episode for {policy_name}")
        ax.set_ylabel("Reject rate")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Iteration")
        ax.legend()
        ax.grid(axis="both")
        ax.set_axisbelow(True)

        figures.append(mo.mpl.interactive(fig))

    return mo.vstack(figures)


@app.cell
def _(filter_exp, reject_data):
    make_reject_plot(reject_data, filter_exp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Rejection rate (summary)""")
    return


@app.cell(hide_code=True)
def _(reject_data):
    _random_exp = next(iter(reject_data))
    _iterations = reject_data[_random_exp].shape[0]
    reject_rate_summary_iters_widget = mo.ui.number(
        start=1, stop=_iterations, debounce=True, label="Latest iterations to consider:"
    )
    reject_rate_summary_iters_widget
    return (reject_rate_summary_iters_widget,)


@app.function
def get_reject_rate_summary(reject_data_exps, latest_iters=10):
    result = {exp_name: {} for exp_name in reject_data_exps.keys()}
    for exp_name in result:
        result[exp_name] = reject_data_exps[exp_name].tail(latest_iters).mean()

    # We transpose because we can the columns to be the policies and rows the experiments.
    df = pd.DataFrame(result).T
    df.sort_index(axis=1, inplace=True)
    return df


@app.cell
def _(reject_data, reject_rate_summary_iters_widget):
    get_reject_rate_summary(reject_data, reject_rate_summary_iters_widget.value)
    return


@app.cell
def _():
    mo.md(r"""## Evaluation data""")
    return


@app.cell
def _():
    mo.md(r"""### Cumulative reward""")
    return


@app.cell
def _(evaluation_data, experiments_df):
    reward_data_eval = {}
    for _exp in experiments_df["name"]:
        reward_data_eval[_exp] = get_reward_data_single_exp(evaluation_data[_exp])
    return (reward_data_eval,)


@app.cell
def _(filter_exp, reward_data_eval):
    make_reward_plot(reward_data_eval, filter_exp, mode="eval")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""#### Reward summary""")
    return


@app.cell
def _(reward_data_eval, reward_summary_iters_widget):
    get_reward_summary(reward_data_eval, reward_summary_iters_widget.value)
    return


@app.cell
def _():
    mo.md(r"""### Reject rate""")
    return


@app.function
def get_reject_data_single_exp_eval(exp_data):
    iterations = len(exp_data)

    # Get agents from a random iteration and a random episode. We assume the agents
    # keeps the same (that's true).
    agents = exp_data[0]["env_runners"]["hist_stats"]["observation_input_rate"][0].keys()

    reject_data = {agent: [] for agent in agents}
    reject_data["all"] = []

    for i in range(iterations):
        episodes = exp_data[i]["env_runners"]["episodes_this_iter"]

        # Support multiple episodes per iteration
        if isinstance(episodes, int):
            num_episodes = episodes
        else:
            num_episodes = len(episodes)

        # Accumulate per agent and all for averaging
        agent_total_reject = {agent: 0 for agent in agents}
        agent_total_obs_input = {agent: 0 for agent in agents}
        total_reject_all, obs_input_rate_all = 0, 0

        for ep_idx in range(num_episodes):
            # Calculate percent for each agent.
            for agent in agents:
                obs_input_rate = np.sum(
                    exp_data[i]["env_runners"]["hist_stats"]["observation_input_rate"][ep_idx][agent]
                )

                action_reject = np.sum(exp_data[i]["env_runners"]["hist_stats"]["action_reject"][ep_idx][agent])
                local_reject = np.sum(
                    exp_data[i]["env_runners"]["hist_stats"]["incoming_rate_local_reject"][ep_idx][agent]
                )
                forward_reject = np.sum(exp_data[i]["env_runners"]["hist_stats"]["forward_reject_rate"][ep_idx][agent])

                total_reject = action_reject + local_reject + forward_reject

                agent_total_reject[agent] += total_reject
                agent_total_obs_input[agent] += obs_input_rate

                total_reject_all += total_reject
                obs_input_rate_all += obs_input_rate

        # Average for each agent over all episodes in iteration
        for agent in agents:
            if agent_total_obs_input[agent] > 0:
                reject_data[agent].append(agent_total_reject[agent] / agent_total_obs_input[agent])
            else:
                reject_data[agent].append(0)

        # Cumulative for all agents.
        if obs_input_rate_all > 0:
            reject_data["all"].append(total_reject_all / obs_input_rate_all)
        else:
            reject_data["all"].append(0)

    return pd.DataFrame(reject_data)


@app.cell
def _(evaluation_data, experiments_df):
    reject_data_eval = {}
    for _exp in experiments_df["name"]:
        reject_data_eval[_exp] = get_reject_data_single_exp_eval(evaluation_data[_exp])
    return (reject_data_eval,)


@app.cell
def _(filter_exp, reject_data_eval):
    make_reject_plot(reject_data_eval, filter_exp, mode="eval")
    return


@app.cell
def _():
    mo.md(r"""#### Reject rate summary""")
    return


@app.cell
def _(reject_data_eval, reject_rate_summary_iters_widget):
    get_reject_rate_summary(reject_data_eval, reject_rate_summary_iters_widget.value)
    return


if __name__ == "__main__":
    app.run()
