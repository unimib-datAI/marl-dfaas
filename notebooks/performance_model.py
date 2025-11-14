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
    mo.md(r"""## Example single execution""")
    return


@app.cell
def _():
    _arrival_rate = 550
    _warm_service_time = 2
    _cold_service_time = 25
    _idle_time_before_kill = 10 * 60

    print("arrival_rate:", _arrival_rate)
    print("warm_service_time:", _warm_service_time)
    print("cold_service_time:", _cold_service_time)
    print("idle_time_before_kill:", _idle_time_before_kill)

    _props1, _props2 = perfmodel.get_sls_warm_count_dist(
        _arrival_rate, _warm_service_time, _cold_service_time, _idle_time_before_kill
    )
    perfmodel.print_props(_props1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Single experiment""")
    return


@app.function
def print_experiment(arrival_rate=100, warm_service_time=2, cold_service_time=25, idle_time_before_kill=10 * 60):
    print("Arguments:")
    print(f"  arrival_rate: {arrival_rate} reqs/s")
    print(f"  warm_service_time: {warm_service_time} s")
    print(f"  cold_service_time: {cold_service_time} s")
    print(f"  idle_time_before_kill: {idle_time_before_kill} s")

    props1, _ = perfmodel.get_sls_warm_count_dist(
        arrival_rate, warm_service_time, cold_service_time, idle_time_before_kill
    )

    print("\nResult:")
    for key in props1:
        print(f"  {key}: {props1[key]}")


@app.function
def run_experiment(arrival_rate=100, warm_service_time=2, cold_service_time=25, idle_time_before_kill=10 * 60):
    props1, _ = perfmodel.get_sls_warm_count_dist(
        arrival_rate, warm_service_time, cold_service_time, idle_time_before_kill
    )

    return props1


@app.cell
def _():
    print_experiment(100, 10, 25, 10 * 60)
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

            result = run_experiment(arrival_rate, warm_service_time, cold_service_time, idle_time_before_kill)
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
