import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import json
    import gzip
    import re

    import marimo as mo

    import networkx as nx
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

            experiments.append(
                {"name": str(name), "exp_dir": str(exp_dir), "entropy": entropy, "entropy_decay": entropy_decay}
            )

        return pd.DataFrame(experiments), experiments_data

    experiments_df, experiments_data = load_experiments(exp_base_dir_widget.path().resolve().absolute())
    return experiments_data, experiments_df


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
        fig = utils.get_figure(f"entropy_coeff")
        ax = fig.subplots()

        for exp_name in entropy_data.keys():
            entropy_start = experiments_df[experiments_df["name"] == exp_name]["entropy"].values[0]
            entropy_decay = experiments_df[experiments_df["name"] == exp_name]["entropy"].values[0]
            label = f"E {entropy_start} D {entropy_decay}"
            ax.plot(entropy_data[exp_name]["entropy_coeff"], label=label)

        ax.set_title(f"Entropy coefficient decay")
        ax.set_ylabel("Entropy coefficient")
        ax.set_xlabel("Iteration")
        ax.legend()
        ax.grid(axis="both")
        ax.set_axisbelow(True)

        return mo.mpl.interactive(fig)

    make_entropy_coeff_plot(experiments_df, entropy_data)
    return


@app.cell
def _(entropy_data):
    def make_entropy_plot(entropy_data):
        # Get the policies by name. We assume all experiments have the same number of
        # policies.
        random_exp = next(iter(entropy_data))
        policy_names = sorted(entropy_data[random_exp].filter(regex="^policy").columns)

        figures = []
        for policy_name in policy_names:
            fig = utils.get_figure(f"entropy_{policy_name}")
            ax = fig.subplots()

            for exp_name in entropy_data.keys():
                ax.plot(entropy_data[exp_name][policy_name], label=exp_name)

            ax.set_title(f"Entropy for {policy_name}")
            ax.set_ylabel("Entropy")
            ax.set_yscale("symlog")
            ax.set_xlabel("Iteration")
            ax.legend()
            ax.grid(axis="both")
            ax.set_axisbelow(True)

            figures.append(mo.mpl.interactive(fig))

        return mo.vstack(figures)

    make_entropy_plot(entropy_data)
    return


if __name__ == "__main__":
    app.run()
