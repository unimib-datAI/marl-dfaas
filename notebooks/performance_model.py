import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np

    import utils
    import perfmodel


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Performance Model

    I use the Python PACSLTK (PACS Lambda ToolKit) module to simulate FaaS processing on a node.

    This is the original source code: https://github.com/pacslab/serverless-performance-modeling
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Single experiment""")
    return


@app.cell(hide_code=True)
def _():
    arrival_rate_widget = mo.ui.number(value=100, label="Arrival rate (reqs/s)", debounce=True)
    warm_service_time_widget = mo.ui.number(value=10, label="Warm service time (seconds)", debounce=True)
    cold_service_time_widget = mo.ui.number(value=25, label="Cold service time (seconds)", debounce=True)
    idle_time_before_kill_widget = mo.ui.number(value=10 * 60, label="Idle time before kill (seconds)", debounce=True)
    max_concurrency_widget = mo.ui.number(value=1000, label="Max concurrency (containers)", debounce=True)
    faster_solution_widget = mo.ui.checkbox(value=True, label="Faster solution")

    mo.vstack(
        [
            arrival_rate_widget,
            warm_service_time_widget,
            cold_service_time_widget,
            idle_time_before_kill_widget,
            max_concurrency_widget,
            faster_solution_widget,
        ]
    )
    return (
        arrival_rate_widget,
        cold_service_time_widget,
        faster_solution_widget,
        idle_time_before_kill_widget,
        max_concurrency_widget,
        warm_service_time_widget,
    )


@app.cell
def _(
    arrival_rate_widget,
    cold_service_time_widget,
    faster_solution_widget,
    idle_time_before_kill_widget,
    max_concurrency_widget,
    warm_service_time_widget,
):
    props1, _ = perfmodel.get_sls_warm_count_dist(
        arrival_rate_widget.value,
        warm_service_time_widget.value,
        cold_service_time_widget.value,
        idle_time_before_kill_widget.value,
        max_concurrency_widget.value,
        faster_solution=faster_solution_widget.value,
    )

    props1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Varying arrival rate""")
    return


@app.cell
def _():
    def run_exp_arrival_rate():
        max_arrival_rate = 150

        rejection_prob = np.zeros(max_arrival_rate + 1)

        warm_service_time = 15
        cold_service_time = 30
        idle_time_before_kill = 10 * 60

        for arrival_rate in mo.status.progress_bar(range(max_arrival_rate + 1)):
            if arrival_rate == 0:
                # Skip basic case.
                continue

            result, _ = perfmodel.get_sls_warm_count_dist(
                arrival_rate, warm_service_time, cold_service_time, idle_time_before_kill
            )
            rejection_prob[arrival_rate] = result["rejection_prob"]

        return rejection_prob

    rejection_prob = run_exp_arrival_rate()
    return (rejection_prob,)


@app.cell
def _(rejection_prob):
    def make_plot(rejection_prob):
        fig = utils.get_figure("arrival_rate")
        ax = fig.subplots()

        ax.plot(rejection_prob)

        ax.set_ylabel("Rejection prob")
        ax.set_ylim(bottom=0, top=1.1)  # Is a probability

        ax.set_xlabel("Arrival rate")

        ax.grid(axis="both")
        ax.set_axisbelow(True)  # By default the axis is over the content.

        return mo.mpl.interactive(fig)

    make_plot(rejection_prob)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Speed

    The `get_sls_warm_count_dist()` function is cached to speed up calls, but this only works for integer values of arrival rate.
    """
    )
    return


@app.cell
def _():
    def speed(trials, rng_seed=42):
        max_arrival_rate = 150
        warm_service_time = 15
        cold_service_time = 30
        idle_time_before_kill = 10 * 60

        rng = np.random.default_rng(rng_seed)
        arrival_rate = rng.integers(low=1, high=max_arrival_rate, endpoint=True, size=trials)

        for i in mo.status.progress_bar(range(trials)):
            perfmodel.get_sls_warm_count_dist(
                arrival_rate[i], warm_service_time, cold_service_time, idle_time_before_kill
            )

    speed(10000)

    perfmodel.get_sls_warm_count_dist.cache_info()
    return


if __name__ == "__main__":
    app.run()
