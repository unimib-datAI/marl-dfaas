"""
This module generates bandwidth traces used to simulate network latency across
links between nodes in the DFaaS environment.

The traces are based from the base file "5G_trace.csv", located in the
"dataset" directory.

It can be executed as a standalone script for trace file generation and
plotting, or imported as a library to generate traces in memory from another
module. For the latter use case, see the "generate_traces" function.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _random_shift_trace(throughput_values, shift_range=(-100, 100), rng=None):
    """Apply a random circular shift to the throughput trace."""
    if rng is None:
        rng = np.random.default_rng()
    shift = rng.integers(shift_range[0], shift_range[1] + 1)
    shifted_trace = np.roll(throughput_values, shift)
    return shifted_trace


def _multiple_random_shifts(throughput_values, num_shifts=5, rng=None):
    """Generate multiple randomly shifted throughput traces."""
    shifted_traces = []
    for i in range(num_shifts):
        shifted = _random_shift_trace(throughput_values, shift_range=(-1000, 1000), rng=rng)
        shifted_traces.append(shifted)
    return shifted_traces


def _invert_trace_middle(throughput_values) -> np.array:
    """Invert the order of the two halves of a trace."""
    n = len(throughput_values)
    middle_idx = n // 2
    # Create inverted trace by swapping halves
    # Second half (middle_idx to end) goes to beginning
    # First half (0 to middle_idx) goes to end
    inverted_trace = np.concatenate(
        [
            throughput_values[middle_idx:],  # Second half to beginning
            throughput_values[:middle_idx],  # First half to end
        ]
    )
    return inverted_trace


def _generate_trace_from_base(base_array, size, random_noise=0.1, rng=None):
    """Generate a new trace by randomly sampling and adding noise."""
    if rng is None:
        rng = np.random.default_rng()

    # Generate random integer indices
    base_length = len(base_array)
    indices = rng.integers(0, base_length, size=size)

    # Generate random percentage changes in a vectorized way
    percentage_changes = rng.uniform(-random_noise, random_noise, size=size)

    # Apply the random percentage changes to the selected elements
    random_values = base_array[indices] * (1 + percentage_changes)
    return np.round(random_values, 2)  # Round and return the new values


def _plot_trace(bw, name, plot_folder, dpi=100):
    """Plot the entire throughput trace and save it to disk using axis object.

    Args:
        bw: The bandwidth trace to plot.
        name: Name of the output plot file (without extension).
        dpi: Plot DPI.
        plot_folder: Folder to save the plot.

    Raises:
        ValueError: If plot_folder is None.
    """
    plt.rcParams.update({"font.size": 21})
    fig, ax = plt.subplots(figsize=(30, 8))
    ax.plot(bw)
    # naming the x axis
    ax.set_xlabel("Timestep")
    # naming the y axis
    ax.set_ylabel("Throughput(Mbps)")
    ax.grid(True)
    fig.tight_layout()
    plot_folder = Path(plot_folder)
    plot_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_folder / f"{name}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_traces(
    base_trace: np.ndarray, num_traces: int, max_len: int, seed: int = 42, random_noise: float = 0.1
) -> list:
    """
    Generate multiple traces from a base trace array.

    Args:
        base_trace (np.ndarray): The original throughput trace.
        num_traces (int): Number of traces to generate.
        max_len (int): Length of each generated trace.
        seed (int): Seed for randomness.
        random_noise (float): Noise range to apply during generation.

    Returns:
        List of np.ndarray traces.
    """
    # Use numpy random generator for reproducibility
    rng = np.random.default_rng(seed)
    # Generate additional traces
    additional_traces = _multiple_random_shifts(base_trace, num_shifts=5, rng=rng)
    additional_traces.append(_invert_trace_middle(base_trace))
    # concatenate
    all_traces = np.concatenate((base_trace, np.concatenate(additional_traces)))
    traces = []
    for i in range(num_traces):
        # For each trace, initialize a new RNG with deterministically different seed
        trace_rng = np.random.default_rng(seed * (i + 1))
        bw = _generate_trace_from_base(all_traces, max_len, random_noise=random_noise, rng=trace_rng)
        traces.append(bw)
    return traces


def main(
    input_folder: Path, output_folder: Path, max_len: int, num_traces: int, seed: int = 42, skip_plots: bool = False
):
    # Load base trace.
    df = pd.read_csv(input_folder / "5G_trace.csv")
    base_trace = df["Throughput"].to_numpy()

    # Generate traces.
    traces = generate_traces(base_trace=base_trace, num_traces=num_traces, max_len=max_len, seed=seed)

    output_folder.mkdir(parents=True, exist_ok=True)

    for i, bw in enumerate(traces):
        # Generate and save the plot only if required.
        if not skip_plots:
            _plot_trace(bw, name=f"5G_trace{i + 1}", plot_folder=output_folder)

        df_out = pd.DataFrame(bw, columns=["Throughput"])
        df_out.to_csv(output_folder / f"5G_trace{i + 1}.csv", index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and manipulate network throughput traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder", type=Path, default=Path("dataset"), help="Input folder containing 5G_trace.csv"
    )
    parser.add_argument(
        "--output-folder", type=Path, default=Path("network_traces/"), help="Folder to save generated traces and plots"
    )
    parser.add_argument("--max-len", type=int, default=288, help="Maximum length of generated trace")
    parser.add_argument("--num-traces", type=int, default=10, help="Number of traces to generate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
    parser.add_argument("--skip-plots", action="store_true", help="If set, skip plot generation (default: False)")
    args = parser.parse_args()

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        max_len=args.max_len,
        num_traces=args.num_traces,
        seed=args.seed,
        skip_plots=args.skip_plots,
    )
