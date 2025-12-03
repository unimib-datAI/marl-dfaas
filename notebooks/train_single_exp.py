import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo

    import networkx as nx
    import pandas as pd
    import numpy as np

    import utils


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Train summary of a single experiment

    This notebook presents plots related to a single experiment. It can display plots and information from two sources:

    1. **Training iterations**: in each iteration, the same number of episodes is played, with a different set of randomly generated seeds each time.

    2. **Evaluation iterations**: same as the training iteration, but the seeds are kept constant across iterations. This is useful to evaluate the agents over time on the same episodes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment loading
    """)
    return


@app.cell(hide_code=True)
def _():
    exp_dir_widget = mo.ui.file_browser(
        initial_path=Path("results"), selection_mode="directory", multiple=False, label="Experiment path: "
    )

    exp_dir_widget
    return (exp_dir_widget,)


@app.cell
def _():
    exp_dir_mode_widget = mo.ui.dropdown(options=["train", "eval"], value="train", label="Experiment mode:")

    exp_dir_mode_widget
    return (exp_dir_mode_widget,)


@app.cell
def _(exp_dir_mode_widget, exp_dir_widget):
    # exp_dir_widget.path() is None at the start of the notebook! So we wait until a directory
    # has been selected.
    # NOTE: to select a directory, click on its icon
    mo.stop(exp_dir_widget.path() is None)

    _exp_dir = exp_dir_widget.path().resolve().absolute()

    if exp_dir_mode_widget.value == "train":
        exp_data = pd.read_json(_exp_dir / "result.json.gz", lines=True, compression="gzip")
    else:
        exp_data = pd.read_json(_exp_dir / "evaluation.json")

    env = utils.get_env(_exp_dir)

    mo.md(f"""
    **Experiment prefix dir**: `{_exp_dir.parent.as_posix()!r}`  
    **Experiment name**:       `{_exp_dir.name!r}`  
    **Agents**:                {env.agents}  
    **Mode**:                  {exp_dir_mode_widget.value}  
    **Iterations**:            {exp_data.shape[0]}
    """)
    return env, exp_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Network topology
    """)
    return


@app.cell(hide_code=True)
def _(env):
    def make_networkx_plot(graph):
        fig = utils.get_figure("network")
        ax = fig.subplots()

        ax.axis("off")

        options = {
            "ax": ax,
            "node_size": 2500,
            "node_color": "white",
            "edgecolors": "black",
        }

        ax.set_axisbelow(True)

        nx.draw_networkx(graph, **options)

        return fig

    make_networkx_plot(env.network)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Loading episodes data
    """)
    return


@app.cell
def _(exp_data):
    def get_episode_data(exp_data, iter_idx, epi_idx):
        # The episode stats are under env_runners -> hist_stats dictionary. Each key is a metric.
        keys = exp_data.iloc[iter_idx]["env_runners"]["hist_stats"]

        # Select only a sub-set of metrics (each metric is a column).
        cols = [
            "observation_input_rate",
            "action_local",
            "action_forward",
            "action_reject",
            "incoming_rate",
            "incoming_rate_reject",
            "incoming_rate_local_reject",
            "forward_reject_rate",
            "reward",
        ]
        extra_cols = [key for key in keys if key.startswith("action_forward_to")]
        columns = cols + extra_cols

        # Extract the metrics/column one by one by creating DataFrames.
        datas = []
        for key in columns:
            data = keys.get(key)
            if data is None:
                # Missing key, maybe it is an old experiment without action_forward_to_XXX.
                continue

            df = pd.DataFrame(data)
            if df.empty:
                continue

            # Each cell is a list (one item for each step). I want to extract items as rows, one for
            # each step (and for each agent).
            exploded = df.apply(pd.Series.explode)

            # Add step column. It will be used to merge DataFrames.
            exploded["step"] = range(1, len(exploded) + 1)

            # We have several columns, one for each agent. Convert the columns as rows under the
            # "agent" column, to just have ["step", "agent", "{key}"] columns.
            final = exploded.melt(id_vars="step", var_name="agent", value_name=key)

            # Keep only {key} column and step and agent as index.
            datas.append(final.set_index(["step", "agent"])[key])

        if not datas:
            # Return empty DataFrame if no valid data.
            return pd.DataFrame(columns=["step", "agent"] + columns)

        # Merge the extracted DataFrames by column. We merge by the custom index (step and agent),
        # but as final result I want a normal index (integer-based).
        result = pd.concat(datas, axis=1).reset_index()
        return result

    @mo.cache
    def get_episodes_data(exp_data):
        """Warning: this is slow!"""
        iterations = exp_data["training_iteration"].max()
        episodes_data = []

        with mo.status.progress_bar(total=iterations, title="Loading episodes data") as bar:
            for iteration in range(iterations):
                episodes = exp_data.iloc[iteration]["env_runners"]["episodes_this_iter"]
                assert episodes == 1
                episodes_data.append(get_episode_data(exp_data, iteration, 1))
                bar.update()

        return episodes_data

    episodes_data = get_episodes_data(exp_data)
    return (episodes_data,)


@app.cell
def _():
    return


@app.cell
def _():
    mo.md(r"""
    ## Reward
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Cumulative reward per episode

    It is the cumulative reward per episode. Since we have one episode per iteration, it is simply the total reward for that episode. The reward of the "all" agent is the sum of the rewards obtained by all agents.
    """)
    return


@app.cell
def _(env, episodes_data):
    def get_reward_data_sum(episodes_data, env):
        """Returns the cumulative reward per episode for each agent and all agents."""
        iters_n = len(episodes_data)
        reward_sum = {agent: np.empty(iters_n) for agent in env.agents}
        reward_sum["all"] = np.empty(iters_n)

        # Iterate over iterations (with its DataFrame).
        for iter_idx, df in enumerate(episodes_data):
            # Group by agent and sum reward for this episode (iteration)
            agent_rewards = df.groupby("agent")["reward"].sum()
            for agent in env.agents:
                reward_sum[agent][iter_idx] = agent_rewards.get(agent, 0.0)

            # Sum across all agents.
            reward_sum["all"][iter_idx] = np.sum([reward_sum[agent][iter_idx] for agent in env.agents])

        return reward_sum

    reward_sum = get_reward_data_sum(episodes_data, env)
    return (reward_sum,)


@app.cell
def _(reward_sum):
    def make_cumulative_reward_plot(reward_sum):
        figures = []
        for agent, reward in sorted(reward_sum.items()):
            fig = utils.get_figure(f"reward_cumulative_{agent}")
            ax = fig.subplots()

            ax.plot(reward)

            moving_average, window_size = utils.get_moving_average(reward)
            ax.plot(moving_average, label=f"Moving average ({window_size} steps)", color="red")

            ax.set_title(f"Average cumulative reward per episode ({agent = })")
            ax.set_ylabel("Reward")
            ax.set_xlabel("Training iteration")

            ax.legend()
            ax.grid(axis="both")
            ax.set_axisbelow(True)  # By default the axis is over the content.

            figures.append(mo.mpl.interactive(fig))

        return mo.vstack(figures)

    make_cumulative_reward_plot(reward_sum)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Average reward per step

    It is the average reward per step for each agent. The special "all" agent represents the average across all agents.
    """)
    return


@app.cell
def _(env, episodes_data):
    def get_average_reward_per_step(episodes_data, env):
        """Returns the average reward per step (not per episode) for each agent and all agents."""
        iters_n = len(episodes_data)
        avg_reward_per_step = {agent: np.empty(iters_n) for agent in env.agents}
        avg_reward_per_step["all"] = np.empty(iters_n)

        # Iterate over iterations (with its DataFrame).
        for iter_idx, df in enumerate(episodes_data):
            # Iterate over agents for the same iteration.
            for agent, agent_data in df.groupby("agent"):
                avg_reward_per_step[agent][iter_idx] = agent_data["reward"].mean()

            # We just average all rewards for this iteration.
            avg_reward_per_step["all"][iter_idx] = df["reward"].mean()

        return avg_reward_per_step

    reward_avg_step = get_average_reward_per_step(episodes_data, env)
    return (reward_avg_step,)


@app.cell
def _(reward_avg_step):
    def make_avg_reward_per_step_plot(avg_reward_per_step):
        figures = []
        for agent, avg_reward in sorted(avg_reward_per_step.items()):
            fig = utils.get_figure(f"avg_reward_per_step_{agent}")
            ax = fig.subplots()

            ax.plot(avg_reward)

            moving_average, window_size = utils.get_moving_average(avg_reward)
            ax.plot(moving_average, label=f"Moving average ({window_size} steps)", color="red")

            ax.set_title(f"Average reward per step ({agent = })")
            ax.set_ylabel("Reward per step")
            ax.set_xlabel("Training iteration")

            ax.legend()
            ax.grid(axis="both")
            ax.set_axisbelow(True)

            figures.append(mo.mpl.interactive(fig))
        return mo.vstack(figures)

    make_avg_reward_per_step_plot(reward_avg_step)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
