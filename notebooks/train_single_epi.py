import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Train summary for a single episode""")
    return


@app.cell
def _():
    # Common imports.
    from pathlib import Path

    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import networkx as nx
    import pandas as pd

    return Path, mo, nx, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Experiment loading""")
    return


@app.cell
def _(Path, mo, nx, pd):
    # Base directory where the experiments are located.
    _prefix_dir = Path("../expand-action-space/results/").resolve().absolute()

    # Experiment directory.
    _exp_dir = _prefix_dir / "DF_20250929_144721_PPO_5_agents_expanded_actions"

    exp_data = pd.read_json(_exp_dir / "result.json", lines=True)

    # FIXME
    _adjlist = ["node_0 node_1 node_2 node_3", "node_1 node_4", "node_2 node_3", "node_3", "node_4"]
    network = nx.parse_adjlist(_adjlist)

    mo.md(f"""
    Experiment prefix dir: {_prefix_dir.as_posix()!r}  
    Experiment name:       {_exp_dir.name!r}  
    Agents:                WIP  
    Mode:                  train  
    Iterations:            {exp_data.shape[0]}
    """)
    return exp_data, network


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Episode selection""")
    return


@app.cell
def _(exp_data, mo):
    # Which iteration (training or evaluation) to select?
    iteration_idx = 4

    # Which episode from the single iteration to select?
    episode_idx = 0

    assert 0 <= iteration_idx <= exp_data.shape[0] - 1, "iteration_idx must be a valid iteration index!"

    assert 0 <= episode_idx < exp_data.iloc[iteration_idx]["env_runners"]["num_episodes"], (
        "episode_idx must be a valid episode index!"
    )

    mo.md(f"""
    Selected iteration:    {iteration_idx}  
    Selected episode:      {episode_idx}
    """)
    return episode_idx, iteration_idx


@app.cell
def _(mo):
    mo.md(r"""### Network topology""")
    return


@app.cell(hide_code=True)
def _(network, nx, plt):
    def make_networkx_plot(graph):
        plt.close(fig=f"networkx")
        fig = plt.figure(num=f"networkx", layout="constrained")
        fig.canvas.header_visible = False
        ax = fig.subplots()

        ax.axis("off")
        ax.set_axisbelow(True)

        # Compute positions for nodes to allow referencing.
        pos = nx.spring_layout(graph)

        options = {
            "ax": ax,
            "node_size": 2500,
            "node_color": "white",
            "edgecolors": "black",
            "pos": pos,
        }

        nx.draw_networkx(graph, **options)

        return fig

    make_networkx_plot(network)
    return


@app.cell
def _(episode_idx, exp_data, iteration_idx, pd):
    def get_input_rate(exp_data, iter_idx, epi_idx):
        iter_data = pd.DataFrame(exp_data.iloc[iter_idx]["env_runners"]["hist_stats"]["observation_input_rate"]).T

        # Convert the single column (list of values) to a series of columns.
        iter_data = iter_data[0].apply(pd.Series).T

        return iter_data

    input_rate = get_input_rate(exp_data, iteration_idx, episode_idx)
    return (input_rate,)


@app.cell
def _(input_rate, mo, plt):
    def make_input_rate_plot(input_rate):
        figures = []
        for agent in input_rate.columns:
            plt.close(fig=f"input_rate_{agent}")
            fig = plt.figure(num=f"input_rate_{agent}", layout="constrained")
            fig.canvas.header_visible = False
            ax = fig.subplots()

            ax.plot(input_rate[agent])

            # moving_average, window_size = base.get_moving_average(reward)
            # ax.plot(moving_average, label=f"Moving average ({window_size} steps)", color="red")

            ax.set_title(f"Input rate per step ({agent = })")
            ax.set_ylabel("Input rate (req/s)")
            ax.set_xlabel("Step")

            ax.grid(axis="both")
            ax.set_axisbelow(True)  # By default the axis is over the content.

            figures.append(mo.mpl.interactive(fig))

        return mo.vstack(figures)

    make_input_rate_plot(input_rate)
    return


@app.cell
def _(pd):
    pd.read_
    return


if __name__ == "__main__":
    app.run()
