import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Train summary for a single episode""")
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    import utils

    return Path, mo, nx, pd, plt, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Experiment loading""")
    return


@app.cell
def _(Path, mo):
    exp_dir_widget = mo.ui.file_browser(
        initial_path=Path("results"), selection_mode="directory", multiple=False, label="Experiment path: "
    )

    exp_dir_widget
    return (exp_dir_widget,)


@app.cell
def _(exp_dir_widget, mo, pd, utils):
    # exp_dir_widget.path() is None at the start of the notebook! So we wait until a directory
    # has been selected.
    mo.stop(exp_dir_widget.path() is None)

    _exp_dir = exp_dir_widget.path().resolve().absolute()

    exp_data = pd.read_json(_exp_dir / "result.json", lines=True)

    env = utils.get_env(_exp_dir)

    mo.md(f"""
    **Experiment prefix dir**: `{_exp_dir.parent.as_posix()!r}`  
    **Experiment name**:       `{_exp_dir.name!r}`  
    **Agents**:                {env.agents}  
    **Mode**:                  train  
    **Iterations**:            {exp_data.shape[0]}
    """)
    return env, exp_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Episode selection""")
    return


@app.cell(hide_code=True)
def _(exp_data, mo):
    _iter_start = 0
    _iter_stop = exp_data.shape[0] - 1

    iteration_idx = mo.ui.number(
        start=_iter_start,
        stop=_iter_stop,
        step=1,
        value=_iter_stop,
        label=f"Iteration index [{_iter_start}, {_iter_stop}]: ",
        debounce=True,
    )

    iteration_idx
    return (iteration_idx,)


@app.cell(hide_code=True)
def _(exp_data, iteration_idx, mo):
    _epi_start = 0
    _epi_stop = exp_data.iloc[iteration_idx.value]["env_runners"]["num_episodes"]

    episode_idx = mo.ui.number(
        start=_epi_start,
        stop=_epi_stop,
        step=1,
        value=_epi_start,
        label=f"Episode index [{_epi_start}, {_epi_stop}]: ",
        debounce=True,
    )

    episode_idx
    return (episode_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Network topology""")
    return


@app.cell(hide_code=True)
def _(env, nx, plt):
    def make_networkx_plot(graph):
        plt.close(fig="networkx")
        fig = plt.figure(num="networkx", layout="constrained")
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

    make_networkx_plot(env.network)
    return


@app.cell
def _(episode_idx, exp_data, iteration_idx, pd):
    def get_input_rate(exp_data, iter_idx, epi_idx):
        iter_data = pd.DataFrame(exp_data.iloc[iter_idx]["env_runners"]["hist_stats"]["observation_input_rate"]).T

        # Convert the single column (list of values) to a series of columns.
        iter_data = iter_data[0].apply(pd.Series).T

        return iter_data

    input_rate = get_input_rate(exp_data, int(iteration_idx.value), int(episode_idx.value))
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
def _(input_rate, mo, utils):
    def make_single_input_rate_plot(input_rate):
        fig = utils.get_figure("single_input_rate")
        ax = fig.subplots()

        for agent in input_rate.columns:
            ax.plot(input_rate[agent], label=agent)

        ax.set_title("Input rates per step")
        ax.set_ylabel("Input rate (req/s)")
        ax.set_xlabel("Step")

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.
        ax.legend()

        return mo.mpl.interactive(fig)

    make_single_input_rate_plot(input_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Episode DataFrame""")
    return


@app.cell
def _(episode_idx, exp_data, iteration_idx, pd):
    def get_episode_data(exp_data, iter_idx, epi_idx):
        from functools import reduce

        keys = exp_data.iloc[iter_idx]["env_runners"]["hist_stats"]

        # Filter the keys that are necessary, exclude all others.
        cols = [
            "observation_input_rate",
            "action_local",
            "action_forward",
            "action_reject",
            "incoming_rate",
            "incoming_rate_reject",
            "incoming_rate_local_reject",
            "forward_reject_rate",
        ]
        extra_cols = [key for key in keys if key.startswith("action_forward_to")]
        columns = cols + extra_cols

        datas = []
        for key in columns:
            try:
                data = pd.DataFrame(exp_data.iloc[iter_idx]["env_runners"]["hist_stats"][key])
            except KeyError:
                # Missing key, maybe it is an old experiment without action_forward_to_XXX.
                continue

            # Convert the original DataFrame (1 row x n agents) to (288 rows x n agents).
            exploded = data.apply(pd.Series.explode)

            # Add the step column.
            exploded["step"] = range(1, len(exploded) + 1)

            # Move the n agents columns to a single columns with the agents value.
            # The resulting DataFrame will have n agents * 288 rows.
            final = exploded.melt(id_vars="step", var_name="agent", value_name=key)

            datas.append(final)

        # Merge all single DataFrames to a single one. Keep the common two columns.
        # Must use outer join because some columns (forward to specific nodes) are missing.
        return reduce(lambda left, right: pd.merge(left, right, on=["step", "agent"], how="outer"), datas)

    episode = get_episode_data(exp_data, int(iteration_idx.value), int(episode_idx.value))
    return (episode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step selection""")
    return


@app.cell(hide_code=True)
def _(episode, mo):
    _start = int(episode["step"].min())
    _end = int(episode["step"].max())

    step = mo.ui.number(
        start=_start, stop=_end, step=1, value=_start, label=f"Step selection [{_start}, {_end}]: ", debounce=True
    )
    step
    return (step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Episode step graph""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Input rate and total reject (with %)""")
    return


@app.cell
def _(env, episode, pd, step):
    def text_reject_rate(episode, env, step):
        step_data = episode[episode["step"] == step]

        rows = []
        for agent in env.agents:
            agent_data = step_data[step_data["agent"] == agent]

            input_rate = agent_data["observation_input_rate"].squeeze()
            action_reject = agent_data["action_reject"].squeeze()
            incoming_rate_local_reject = agent_data["incoming_rate_local_reject"].squeeze()
            forward_reject = agent_data["forward_reject_rate"].squeeze()

            total_reject = action_reject + incoming_rate_local_reject + forward_reject

            rows.append({"agent": agent, "input_rate": input_rate, "total_reject": total_reject})

        stats = pd.DataFrame(rows)

        # Calculate the total for all agents.
        all_input_rate = stats["input_rate"].sum()
        all_total_reject = stats["total_reject"].sum()
        all = pd.DataFrame([{"agent": "all", "input_rate": all_input_rate, "total_reject": all_total_reject}])

        stats = pd.concat([all, stats], ignore_index=True)

        # Compute the reject_percent column, set zero if input_rate is zero.
        stats["reject_percent"] = stats.apply(
            lambda row: row["total_reject"] / row["input_rate"] if row["input_rate"] != 0 else 0, axis=1
        )

        return stats

    text_reject_rate(episode, env, step.value)
    return


@app.cell
def _(episode):
    step_data = episode[episode["step"] == 1]
    agent_data = step_data[step_data["agent"] == "node_0"]

    incoming_rate_reject = agent_data["incoming_rate_reject"].iloc[0]

    incoming_rate_reject
    return


@app.cell
def _(env, episode, mo, step):
    def draw_step_graph(episode, env, step):
        step_data = episode[episode["step"] == step]

        import graphviz

        # Create a directed graph using graphviz.
        dot = graphviz.Digraph()

        dot.attr("node", shape="circle", style="filled", fillcolor="white", color="black")

        # Add data as arrows.
        for agent in env.agents:
            agent_data = step_data[step_data["agent"] == agent]

            # Basic action and local processing/reject
            action_local = str(agent_data["action_local"].iloc[0])
            action_forward = str(agent_data["action_forward"].iloc[0])
            action_reject = str(agent_data["action_reject"].iloc[0])
            incoming_rate_reject = agent_data["incoming_rate_reject"].iloc[0]
            incoming_rate_processed = str(agent_data["incoming_rate"].iloc[0] - incoming_rate_reject)
            incoming_rate_reject = str(incoming_rate_reject)

            # Agent node.
            agent_label = f"{agent}\n⬇️{action_local} ↪️{action_forward} 🗑️{action_reject}\n✅{incoming_rate_processed} ❌{incoming_rate_reject}"
            dot.node(agent, label=agent_label)

            # Input rate with dummy nodes.
            dot.node(f"input_{agent}", shape="point")
            dot.edge(f"input_{agent}", agent, label=str(agent_data["observation_input_rate"].iloc[0]))

            # Specific forward to neighbors.
            try:
                for neighbor in env.network.adj[agent]:
                    rate = str(agent_data[f"action_forward_to_{neighbor}"].squeeze())
                    dot.edge(agent, neighbor, label=rate)
            except KeyError:
                # Missing action_forward_to_XXX, fallback to previous version.
                # Note that action_forward is a float represented as str...
                portion = str(round(float(action_forward) / len(env.network.adj[agent])))
                for neighbor in env.network.adj[agent]:
                    dot.edge(agent, neighbor, label=portion)

        # Render to SVG in-memory and display in Marimo
        svg = dot.pipe(format="svg").decode("utf-8")
        return mo.Html(svg).center()

    draw_step_graph(episode, env, step.value)
    return


if __name__ == "__main__":
    app.run()
