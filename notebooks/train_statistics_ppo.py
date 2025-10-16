import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo

    import networkx as nx
    import pandas as pd

    import utils


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Train statistics of a single experiment (PPO)

    This notebook presents training statistics for the policies that have been trained with the PPO algorithm.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Experiment loading""")
    return


@app.cell(hide_code=True)
def _():
    exp_dir_widget = mo.ui.file_browser(
        initial_path=Path("results"), selection_mode="directory", multiple=False, label="Experiment path: "
    )

    exp_dir_widget
    return (exp_dir_widget,)


@app.cell
def _(exp_dir_widget):
    # exp_dir_widget.path() is None at the start of the notebook! So we wait until a directory
    # has been selected.
    mo.stop(exp_dir_widget.path() is None)

    _exp_dir = exp_dir_widget.path().resolve().absolute()

    exp_data = pd.read_json(_exp_dir / "result.json.gz", lines=True, compression="gzip")

    env = utils.get_env(_exp_dir)

    mo.md(f"""
    **Experiment prefix dir**: `{_exp_dir.parent.as_posix()!r}`  
    **Experiment name**:       `{_exp_dir.name!r}`  
    **Agents**:                {env.agents}   
    **Iterations**:            {exp_data.shape[0]}
    """)
    return env, exp_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Network topology""")
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
    mo.md(r"""### Loading training data""")
    return


@app.function
@mo.cache
def get_training_data(exp_data):
    """Returns the training data from all iterations as a DataFrame.

    The DataFrame has one row per policy and per iteration (see "policy_name"
    and "iteration" columns).
    """
    # We first collect the data for each policy for each iteration as multiple rows.
    training_data = []

    iterations = exp_data["training_iteration"].max()
    with mo.status.progress_bar(total=iterations, title="Loading episodes data") as bar:
        for iteration in range(iterations):
            # The training stats are under the info -> learner dictionary, one key for each
            # policy (one policy for each agent).
            policies = exp_data.iloc[iteration]["info"]["learner"]
            for policy_name in policies.keys():
                # We just copy the dict with all stats.
                policy_stats = policies[policy_name]["learner_stats"]

                # Add policy name and iteration, these will be useful columns in the
                # resulting DataFrame.
                policy_stats["policy_name"] = policy_name
                policy_stats["iteration"] = iteration

                training_data.append(policy_stats)

            bar.update()

    training_df = pd.DataFrame(training_data)

    # Move the columns "iteration" and "policy_name" as firsts. This is just for
    # visualization as raw DataFrame.
    cols_to_front = ["iteration", "policy_name"]
    other_cols = [col for col in training_df.columns if col not in cols_to_front]
    new_order = cols_to_front + other_cols
    training_df = training_df[new_order]

    # Remove unused column (always set to 0).
    training_df = training_df.drop("allreduce_latency", axis=1)

    return training_df


@app.cell
def _(exp_data):
    training_data = get_training_data(exp_data)
    return (training_data,)


@app.function
def make_stats_plot(training_data, stats_key, stats_name, stats_ylabel, policy_name=None):
    """Plots training statistics of the given stats_key for all policies.

    If `policy_name` is provided, only this policy is plotted.

    All plots are wrapped inside mo.mpl.interactive().

    Args:
        training_data (pandas.DataFrame): DataFrame containing training statistics.
        stats_key (str): Column name containing the statistic to plot.
        stats_name (str): Name of the statistic for the plot title.
        stats_ylabel (str): Label for the Y-axis.
        policy_name (str, optional): Policy name to plot. By default is None.

    Returns:
        matplotlib.figure.Figure or mo.HTML:
            Single figure or vertically stacked of figures depending on `policy_name`.
    """
    figures = []
    policy_names = sorted(training_data["policy_name"].unique())
    if policy_name is not None:
        policy_names = [policy_name]

    for _policy_name in policy_names:
        fig = utils.get_figure(f"{stats_key}_{_policy_name}")
        ax = fig.subplots()

        # Get stat for a single policy for all iterations. We must reset
        # the index of the resulting Series because the default index is taken
        # from the DataFrame, but in this we have multiple rows for a single
        # iterations!
        loss = training_data.loc[training_data["policy_name"] == _policy_name, stats_key].reset_index(drop=True)

        ax.plot(loss)

        ax.set_title(f"{stats_name} for {_policy_name}")
        ax.set_ylabel(stats_ylabel)
        ax.set_xlabel("Iteration")

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        figures.append(mo.mpl.interactive(fig))

    if policy_name is not None:
        return figures[0]

    return mo.vstack(figures)


@app.cell
def _():
    mo.md(r"""## Stats""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Total Loss

    The overall loss is used to update the policy network in a single gradient step. It is a combination of:

    * Policy loss,
    * Value function loss,
    * Differential Entropy,
    * KL divergence penalty,
    * Entropy coefficient.

    The range depends on the reward scale.
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "total_loss", "Total loss", "Loss")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Policy loss

    This is the loss associated with the policy (actor) network.
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "policy_loss", "Policy loss", "Loss")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Value function Loss

    This is the loss associated with the value (critic) network. It measures how closely the predictions of the value network match the actual returns observed during training.
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "vf_loss", "Value function loss", "Loss")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Value Function Explained Variance

    This is a normalised measure of how well the value function's predictions explain the variation in actual returns. The typical range is from negative to 1. Values should be closer to 1.

    * 1: perfect prediction.
    * 0: the predictions are no better than the mean of the targets.
    * <0: predictions are worse than just using the mean
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "vf_explained_var", "Value Function Explained Variance", "Variance")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Differential Entropy

    /// attention | Warning!

    In the context of a continuous distribution like [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution#Entropy), the entropy is actually the [differential entropy](https://en.wikipedia.org/wiki/Differential_entropy). The differential entropy [is not](https://github.com/pytorch/pytorch/issues/152845#issuecomment-2860403912) the level of exploration like for discrete probabilities distributions.

    ///

    The differential entropy of the policy measures information in terms of probability density. The output of the actor is the concentration parameters for a Dirichlet distribution, and the final action is sampled from this distribution.

    * **Very negative values** indicate that the density is high in a small volume, meaning the **distribution is highly concentrated**.
    * **Values closer to zero** or positive indicate a more diffuse distribution, meaning the **density is lower but spread over a larger volume**.

    Also:

    * If one or more parameters are small, the distribution is concentrated near the corners of the simplex, so the differential entropy is negative.
    * If one or more parameters are larger, the distribution spreads toward the center of the simplex, and the differential entropy increases, meaning the distribution is more spread out.

    Note that the plot shows the average differential entropy for each training iteration.
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "entropy", "Differential Entropy", "Entropy")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### KL divergence

    The KL divergence measures how much the new policy has changed compared to the old policy.

    * A higher mean KL means the policy is changing a lot in one update (possibly too much).
    * A lower mean KL means the policy is not changing much (possibly learning too slowly).

    It is a non-negative metric.
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "kl", "KL divergence", "KL divergence")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Global Gradient Norm

    The Global Gradient Norm is the Euclidean norm of the gradients computed during a single optimisation step. It measures the size of the policy/value gradients during learning.

    * A very large value can indicate unstable learning or exploding gradients.
    * A very small value (close to zero) means the model's parameters are barely changing (possibly due to vanishing gradients or convergence).

    The scale of the value is influenced by the network's structure and the distribution of actions. Fluctuations are normal.
    """
    )
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "grad_gnorm", "Global Gradient Norm", "Global Gradient Norm")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Entropy Coefficient""")
    return


@app.cell
def _(training_data):
    make_stats_plot(training_data, "entropy_coeff", "Entropy Coefficient", "Entropy Coefficient")
    return


if __name__ == "__main__":
    app.run()
