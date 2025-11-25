"""Configuration module for the DFaaS multi-agent environment.

This module provides DFaaSConfig, a class that follows the build pattern to
create, validate and enrich the configuration for the DFaaS environment.

I took inspiration from the PPOConfig class defined in Ray RLLib (2.X old
stack).

This module can also be called as script to read/write configurations. Call it
with "--help" flag for more information.
"""

from pathlib import Path
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats

from dfaas_env import DFaaS
from bandwidth_generator import generate_traces


class DFaaSConfig:
    """Defines a configuration class from which a DFaaS env can be built.

    See the __init__ method for the specific parameters.

    Example usage:
        from dfaas_env_config import DFaaSConfig

        # Create an env using the default configuration.
        env_config = DFaaSConfig()
        env = env_config.build()

        # Export the configuration as dict (safe to write as YAML/JSON).
        config_dict = env_config.to_dict()

        # Load a configuration from a dict. When you load with from_dict(), some
        # parameters if not provided they will be automatically generated.
        config = {"network": ["node_0 node_1", "node_1 node_2"]}
        env_config = DFaaSConfig.from_dict(config)

        # Do not call manually _validate()! It will be called automatically on
        # build().
        env = env_config.build()  # Also validate the config!

        # If you created a DFaaSConfig with from_dict(), make sure to export as
        # dict AFTER you called build().
        config = {"network": ["node_0 node_1", "node_1 node_2"]}
        env_config = DFaaSConfig.from_dict(config)
        env = env_config.build()
        config_dict = env_config.to_dict()  # Safe to write as YAML/JSON.
    """

    def __init__(self):
        """Initialize a DFaaSConfig instance with default values."""

        # Note: the default parameters are taken from this article: "QoS-aware
        # offloading policies for serverless functions in the Cloud-to-Edge
        # continuum" of G. Russo Russo, D. Ferrarelli, D. Pasquali et al. DOI:
        # https://doi.org/10.1016/j.future.2024.02.019

        # Implementation note: when you add a new config parameter, make sure to
        # add the default values here (inside a _X attribute) and then update
        # _validate() and build().
        #
        # If the config parameter depends on some other values (like network
        # nodes), you may want also to update from/to_dict() and build(). See
        # how self.network_links is implemented for more information.

        # Network structure given as Networkx's adjacency list.
        #
        # If you changhe this, you may want also to update "perfmodel_params",
        # "network_links" and "node_ram_gb". Note: if you use from_dict() and do
        # not provide these parameters, they will be generated with default
        # values.
        self.network = ["node_0 node_1"]

        # Internal network representation as NetworkX graph (field used
        # internally to pass to the environment).
        self._network = nx.parse_adjlist(self.network)

        # Number of steps for an episode.
        self.max_steps = 288

        # Input rate generation configuration.
        self.input_rate_method = "synthetic-sinusoidal"

        # All nodes have the same input rate generation method.
        self.input_rate_same_method = True

        # Environment mode.
        self.evaluation = False

        # Path to the bandwidth 5G base trace, used to generate the bandwidth
        # traces. This field is used only if at least one link uses the
        # "generated" bandwidth generation method.
        self.bandwidth_base_trace_path = Path("dataset/5G_trace.csv")

        # Default link parameters value for a single link in the network.
        #
        # The parameters are:
        #
        # - access_delay_ms: Access delay in milliseconds.
        #
        # - bandwidth_mbps_method: Method for bandwidth generation
        # ("static-single", "static-trace" or "generated").
        #
        # - bandwidth_mbps_mean: Mean bandwidth value in Mbps, used for static
        # or generated methods. Ignored if using "static-trace" method.
        #
        # - bandwidth_mbps_random_noise: Random noise factor for bandwidth trace
        # generation (range in [0, 1]). Ignored if using static methods.
        #
        # - bandwidth_mbps: Actual bandwidth trace, of value for each
        # environment step (list of size max_steps). Ignored if using
        # "static-single" or "generated" methods.
        #
        # The bandwidth trace can be given in three ways:
        #
        # 1. Use "generated" method, and it will be automatically generated
        # around the given mean and with the given random noise. Values are
        # rounded to integers.
        #
        # 2. Use "static-single" method, and the given mean value will be
        # repeated for all steps.
        #
        # 3. Use "static-trace" method and give directly the bandwidth trace in
        # bandwidth_mbps.
        #
        # Note: for the generation method the base trace is read from
        # bandwidth_base_trace_path configuration option.
        self._network_links_params = {
            "access_delay_ms": 5,
            "bandwidth_mbps_method": "generated",
            "bandwidth_mbps_mean": 100,
            "bandwidth_mbps_random_noise": 0.1,
            "bandwidth_mbps": None,
        }

        # Network link parameters for each link in the network.
        #
        # Note: since the network links are undirected, you can specify only a
        # direction of the link.
        self.network_links = {"node_0": {"node_1": self._network_links_params.copy()}}

        # Data input size (in bytes) of the single function. The actual value
        # "request_input_data_size_bytes" is extracted from a truncated normal
        # distribution with the given range (inclusive) and given mean and
        # standard deviation.
        #
        # To have a fixed value, just put the range [min, min] and mean/std to
        # [min, 0].
        self.request_input_data_size_bytes_range = [100, 5242880]
        self.request_input_data_size_bytes_mean_std = [1024.0, 1024.0]
        self.request_input_data_size_bytes = None

        # Memory demand (in MB) for the single function. The demand value
        # "request_memory_mb" is extracted uniformly from the given (inclusive)
        # range.
        #
        # To have a single value, just put the range to [min, min].
        self.request_memory_mb_range = [128, 1024]
        self.request_memory_mb = None

        # Warm and cold service time (in ms) for the single function. The
        # values "warm_service_time" and "cold_service_time" are extracted
        # uniformly from the given (inclusive) range. On extraction is guarantee
        # that warm_service_time < cold_service_time.
        #
        # To have a single value, just put the range to [min, min].
        self.warm_service_time_range = [100, 500]
        self.cold_service_time_range = [250, 750]
        self.warm_service_time = None
        self.cold_service_time = None

        # Default RAM capacity in GB for a node.
        self._node_ram_gb_default = 4

        # RAM capacity in GB for each node.
        self.node_ram_gb = {node: self._node_ram_gb_default for node in self._network.nodes}

        # Performance model default parameters for a node in the network.
        #
        # The parameters are:
        #
        # - warm_service_time: Service time for warm function invocation
        # in milliseconds.
        #
        # - cold_service_time: Cold service time in milliseconds.
        #
        # - idle_time_before_kill: Time before idle container is killed in
        # milliseconds.
        #
        # - maximum_concurrency: Maximum number of concurrent requests a node
        # can handle.
        #
        # Note: the parameters (except idle_time_before_kill) should be set to
        # None to allow to be calculated on build(). For maximum_concurrency,
        # the calculated value considers request_memory_mb and node_ram_gb. For
        # warm/cold_service_time, the value considers
        # warm/cold_service_time_range.
        self._perfmodel_params = {
            "warm_service_time": None,
            "cold_service_time": None,
            "idle_time_before_kill": 30,
            "maximum_concurrency": None,
        }

        # Performance model parameters for each node in the network.
        self.perfmodel_params = {"node_0": self._perfmodel_params.copy(), "node_1": self._perfmodel_params.copy()}

        # Seed used when building the environment. Warning: it is not used
        # to reset the environment!
        self.build_seed = 42

    def _validate(self, skip_generated=False):
        """Validate configuration. Automatically called by build().

        If skip_generated is True, the generated config values are skipped.

        Raise ValueError, TypeError or KeyError on validation error."""
        # Generate the network object, it will be used to check nodes and edges.
        self._network = nx.parse_adjlist(self.network)

        # self.max_steps validation.
        if not isinstance(self.max_steps, int):
            raise TypeError(f"max_steps must be int, got {type(self.max_steps)}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        # self.request_input_data_size_bytes_range validation.
        if len(self.request_input_data_size_bytes_range) != 2:
            raise ValueError("request_input_data_size_bytes_range must be [min, max]")
        if self.request_input_data_size_bytes_range[0] < 0:
            raise ValueError("request_input_data_size_bytes_range: min must be non negative")
        if self.request_input_data_size_bytes_range[1] < 0:
            raise ValueError("request_input_data_size_bytes_range: max must be non negative")
        if self.request_input_data_size_bytes_range[0] > self.request_input_data_size_bytes_range[1]:
            raise ValueError("request_input_data_size_bytes_range: min > max")

        # self.request_input_data_size_bytes_mean_std validation.
        if len(self.request_input_data_size_bytes_mean_std) != 2:
            raise ValueError(
                f"request_input_data_size_bytes_mean_std must be [mean, std], got {len(self.request_input_data_size_bytes_mean_std)} values"
            )
        if self.request_input_data_size_bytes_mean_std[0] < 0:
            raise ValueError("request_input_data_size_bytes_mean_std: mean must be non negative")
        if self.request_input_data_size_bytes_mean_std[1] < 0:
            raise ValueError("request_input_data_size_bytes_mean_std: std must be non negative")

        # self.request_input_data_size_bytes validation.
        if not skip_generated:
            if self.request_input_data_size_bytes is None:
                raise TypeError("request_input_data_size_bytes should not be None")

        # self.request_memory_mb_range validation.
        if len(self.request_memory_mb_range) != 2:
            raise ValueError("request_memory_mb_range must be [min, max]")
        if self.request_memory_mb_range[0] < 0:
            raise ValueError("request_memory_mb_range: min must be non negative")
        if self.request_memory_mb_range[1] < 0:
            raise ValueError("request_memory_mb_range: max must be non negative")
        if self.request_memory_mb_range[0] > self.request_memory_mb_range[1]:
            raise ValueError("request_memory_mb_range: min > max")

        if not skip_generated:
            # self.request_memory_mb validation.
            if self.request_memory_mb is None:
                raise TypeError("request_memory_mb should not be None")

        # self.warm_service_time_range validation.
        if len(self.warm_service_time_range) != 2:
            raise ValueError("warm_service_time_range must be [min, max]")
        if self.warm_service_time_range[0] < 0:
            raise ValueError("warm_service_time_range: min must be non negative")
        if self.warm_service_time_range[1] < 0:
            raise ValueError("warm_service_time_range: max must be non negative")
        if self.warm_service_time_range[0] > self.warm_service_time_range[1]:
            raise ValueError("warm_service_time_range: min > max")

        if not skip_generated:
            # self.warm_service_time validation.
            if self.warm_service_time is None:
                raise TypeError("warm_service_time should not be None")

        # self.cold_service_time_range validation.
        if len(self.cold_service_time_range) != 2:
            raise ValueError("cold_service_time_range must be [min, max]")
        if self.cold_service_time_range[0] < 0:
            raise ValueError("cold_service_time_range: min must be non negative")
        if self.cold_service_time_range[1] < 0:
            raise ValueError("cold_service_time_range: max must be non negative")
        if self.cold_service_time_range[0] > self.cold_service_time_range[1]:
            raise ValueError("cold_service_time_range: min > max")

        if not skip_generated:
            # self.cold_service_time validation.
            if self.cold_service_time is None:
                raise TypeError("cold_service_time should not be None")

        # Validate perfmodel_params for all nodes.
        if self.perfmodel_params is None:
            raise ValueError("network_links should not be None")
        perfmodel_keys = set(self._perfmodel_params.keys())
        for node in self._network.nodes:
            # Node exists?
            if node not in self.perfmodel_params:
                raise ValueError(f"perfmodel_params: node {node!r} missing")

            # Node has all keys?
            node_perfmodel_keys = self.perfmodel_params[node].keys()
            if perfmodel_keys != node_perfmodel_keys:
                raise ValueError(f"perfmodel_params: node {node!r} has different set of keys!")

            if skip_generated:
                # Skip single key check for now, even if idle_time_before_kill
                # is not generated.
                continue

            # All required keys have the right values?
            warm_sv_time = self.perfmodel_params[node]["warm_service_time"]
            cold_sv_time = self.perfmodel_params[node]["cold_service_time"]
            idle_time = self.perfmodel_params[node]["idle_time_before_kill"]
            max_conc = self.perfmodel_params[node]["maximum_concurrency"]
            if warm_sv_time is None:
                raise TypeError(f"perfmodel_params: node {node!r}: warm_service_time cannot be None")
            if warm_sv_time <= 0:
                raise ValueError(
                    f"perfmodel_params: node {node!r}: warm_service_time must be positive, got {warm_sv_time}"
                )
            if cold_sv_time is None:
                raise TypeError(f"perfmodel_params: node {node!r}: cold_service_time cannot be None")
            if cold_sv_time <= 0:
                raise ValueError(
                    f"perfmodel_params: node {node!r}: cold_service_time must be positive, got {cold_sv_time}"
                )
            if cold_sv_time < warm_sv_time:
                raise ValueError(f"perfmodel_params: node {node!r}: cold_service_time < warm_service_time")
            if idle_time < 0:
                raise ValueError(f"perfmodel_params: node {node!r}: idle_time must be non-negative, got {idle_time}")
            if not isinstance(max_conc, int):
                raise TypeError(
                    f"perfmodel_params: node {node!r}: maximum_concurrency must be of type int, got {type(max_conc)}"
                )
            if max_conc <= 0:
                raise ValueError(
                    f"perfmodel_params: node {node!r}: maximum_concurrency must be non-negative, got {max_conc}"
                )

        # Validate node_ram_gb for all nodes.
        if self.node_ram_gb is None:
            raise ValueError("node_ram_gb should not be None")
        for node in self._network.nodes:
            # Node exists?
            if node not in self.node_ram_gb:
                raise ValueError(f"node_ram_gb: node {node!r} missing")

            # Node has the right value?
            if self.node_ram_gb[node] <= 0:
                raise ValueError(f"node_ram_gb: node {node!r}: must be positive, got {self.node_ram_gb[node]}")

        # Validate network_links for all edges.
        if self.network_links is None:
            raise ValueError("network_links should not be None")
        seen = {}
        link_params_keys = set(self._network_links_params.keys())
        for u, v in self._network.edges():
            if (u, v) or (v, u) in seen:
                # This edge is already validated, since the graph is undirected.
                continue

            # Search the edge in network_links. Try both (u, v) and (v, u).
            link_params = None
            try:
                link_params = self.network_links[u][v]
            except KeyError:
                pass
            if link_params is None:
                try:
                    link_params = self.network_links[u][v]
                except KeyError:
                    raise KeyError(f"network_links: undirected link ({u}, {v}) not found")

            # The edge has all parameters?
            if link_params_keys != set(link_params.keys()):
                raise ValueError(f"network_links: undirected link ({u}, {v}) has different set of keys")

            # The edge has right values?
            if link_params["access_delay_ms"] < 0:
                raise ValueError(f"network_links: undirected link ({u}, {v}): access_delay_ms must be non-negative")
            if link_params["bandwidth_mbps_mean"] <= 0:
                raise ValueError(f"network_links: undirected link ({u}, {v}): bandwidth_mbps_mean must be non-negative")
            if not 0.0 <= link_params["bandwidth_mbps_random_noise"] <= 1.0:
                raise ValueError(
                    f"network_links: undirected link ({u}, {v}): bandwidth_mbps_random_noise must be in [0, 1]"
                )
            if link_params["bandwidth_mbps_method"] not in ["static-single", "static-trace", "generated"]:
                raise ValueError(f"network_links: undirected link ({u}, {v}): unrecognized bandwidth_mbps_method")

            if not skip_generated:
                if link_params["bandwidth_mbps"] is None:
                    raise ValueError(f"network_links: undirected link ({u}, {v}): bandwidth_mbps should not be None")
                if len(link_params["bandwidth_mbps"]) != self.max_steps:
                    raise ValueError(
                        f"network_links: undirected link ({u}, {v}): bandwidth_mbps trace has wrong length"
                    )

            seen.add((u, v), (v, u))

        if self.build_seed is None:
            raise TypeError("build_seed should not be None")

    def build(self):
        """Build the DFaaS environment from this configuration.

        Return an initialized (but not started) DFaaS environment."""
        # Generate NetworkX graph (internal attribute).
        self._network = nx.parse_adjlist(self.network)

        # network_links may be none if the user give an update network but no
        # network_links.
        if self.network_links is None:
            self.network_links = {}
            for u, v in self._network.edges:
                self.network_links.setdefault(u, {})
                self.network_links[u][v] = self._network_links_params.copy()

        # Same as network_links, see above comment.
        if self.perfmodel_params is None:
            self.perfmodel_params = {}
            for u in self._network.nodes:
                self.perfmodel_params[u] = self._perfmodel_params.copy()

        # Same as network_links, see above comment.
        if self.node_ram_gb is None:
            self.node_ram_gb = {u: self._node_ram_gb_default for u in self._network.nodes}

        # Run a first validation skipping the generated values, since we
        # generate these now.
        self._validate(skip_generated=True)

        rng = np.random.default_rng(seed=self.build_seed)

        # Generate "request_input_data_size_bytes".
        mean, std = self.request_input_data_size_bytes_mean_std
        min_val, max_val = self.request_input_data_size_bytes_range
        # Normalized bounds for truncnorm.
        #
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        self.request_input_data_size_bytes = int(scipy.stats.truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))

        # Generate "request_memory_mb".
        min_mem, max_mem = self.request_memory_mb_range
        self.request_memory_mb = int(rng.integers(min_mem, high=max_mem, endpoint=True))

        # Generate "warm_service_time" and "cold_service_time".
        # We need to guarantee warm_service_time < cold_service_time.
        min_warm, max_warm = self.warm_service_time_range
        min_cold, max_cold = self.cold_service_time_range
        while True:
            warm_service_time = int(rng.integers(min_warm, high=max_warm, endpoint=True))
            cold_service_time = int(rng.integers(min_cold, high=max_cold, endpoint=True))
            if warm_service_time <= cold_service_time:
                self.warm_service_time = warm_service_time
                self.cold_service_time = cold_service_time
                break

        # Calculate maximum_concurrency for each node.
        for node in self._network.nodes:
            # Set warm and cold service time for each node.
            self.perfmodel_params[node]["warm_service_time"] = self.warm_service_time
            self.perfmodel_params[node]["cold_service_time"] = self.cold_service_time

            max_concurrency = int(np.floor((self.node_ram_gb[node] * 1024) / self.request_memory_mb))
            self.perfmodel_params[node]["maximum_concurrency"] = max_concurrency

        # Generate bandwidth trace.
        if self.bandwidth_base_trace_path:
            # Load base trace if provided.
            df = pd.read_csv(self.bandwidth_base_trace_path)
            base_trace = df["Throughput"].to_numpy()
        else:
            base_trace = None

        # Count how many links want to generate the bandwidth trace.
        generated = 0
        random_noise, target_mean = None, None
        for u, v in self._network.edges():
            # Search the link in self.network_links. We may swap (u, v) since
            # the graph is undirected.
            params = None
            try:
                params = self.network_links[u][v]
            except KeyError:
                pass
            if not params:
                try:
                    params = self.network_links[v][u]
                    u, v = v, u
                except KeyError:
                    raise KeyError(f"network_links: undirected link ({u}, {v}) not found")

            match params["bandwidth_mbps_method"]:
                case "generated":
                    generated += 1

                    # FIXME: We currently support only one random noise factor
                    # value, that must be the same for all links. So we need to
                    # check manually all values to be the same.
                    #
                    # The same also for the mean.
                    if not random_noise:
                        random_noise = params["bandwidth_mbps_random_noise"]
                    elif random_noise != params["bandwidth_mbps_random_noise"]:
                        raise ValueError(
                            f"network_links: undirected link ({u}, {v}): bandwidth_mbps_random_noise must be the same value for all links (known limitation)"
                        )
                    if not target_mean:
                        target_mean = params["bandwidth_mbps_mean"]
                    elif target_mean != params["bandwidth_mbps_mean"]:
                        raise ValueError(
                            f"network_links: undirected link ({u}, {v}): bandwidth_mbps_mean must be the same value for all links (known limitation)"
                        )

                case "static-single":
                    # This link has set the "static" method and provided just a
                    # single value: expand to a static trace.
                    params["bandwidth_mbps"] = np.full(
                        self.max_steps, fill_value=params["bandwidth_mbps_mean"]
                    ).tolist()

                case "static-trace":
                    # This link has set manually a complete trace. Use as is.
                    pass

        if generated > 0:
            if not self.bandwidth_base_trace_path:
                raise ValueError("bandwidth_base_trace_path needed but not given")
            if base_trace is None:
                raise ValueError("base_trace should not be none")

            # Generate the bandwidth traces.
            traces = generate_traces(
                base_trace=base_trace,
                num_traces=generated,
                max_len=self.max_steps,
                random_noise=random_noise,
                seed=self.build_seed,
                target_mean=target_mean,
            )

            i = 0
            # Now assign a trace to each link.
            for u, v in self._network.edges():
                # Search the link in self.network_links. We may swap (u, v) since
                # the graph is undirected.
                params = None
                try:
                    params = self.network_links[u][v]
                except KeyError:
                    pass
                if not params:
                    try:
                        params = self.network_links[v][u]
                        u, v = v, u
                    except KeyError:
                        raise KeyError(f"network_links: undirected link ({u}, {v}) not found")

                if params["bandwidth_mbps_method"] == "generated":
                    # Force integer values, not floating point values.
                    params["bandwidth_mbps"] = np.round(traces[i]).astype(int)
                    i += 1

        # Create and return the DFaaS env. Note in its init method it will
        # validate the config.
        return DFaaS(config=self)

    def to_dict(self):
        """Convert DFaaSConfig to a dict, safe to serialize as JSON or YAML.

        WARNING: if you created a modified DFaaSConfig class, eg. with
        from_dict(), make sure to export as dict AFTER you call build(), because
        build() updates/generates some configuration parameters! Call to_dict()
        without build() only if you want the default configuration."""
        result = {}

        # Iterate over all object's attributes (only variables).
        for attr, value in vars(self).items():
            if attr.startswith("_"):
                continue

            match attr:
                case "bandwidth_base_trace_path":
                    # Must convert Path to plain string.
                    result[attr] = value.as_posix()

                case "network_links":
                    # Must convert manually bandwidth_mbps (NumPy arrays) to
                    # plain lists.
                    result[attr] = deepcopy(value)
                    for src, dests in result[attr].items():
                        for dst, params in dests.items():
                            # "bandwidth_mbps" may be not yet generated/set!
                            if params["bandwidth_mbps"] is not None:
                                params["bandwidth_mbps"] = params["bandwidth_mbps"].tolist()

                case _:
                    result[attr] = value

        return result

    @classmethod
    def from_dict(cls, config_dict):
        """Create a DFaaSConfig from a dict."""
        obj = cls()

        if config_dict is None:
            return obj

        # Iterate over given dictionary and assign to the DFaaSConfig.
        # Validation will be done on build().
        network_changed = False
        perfmodel_params_changed = False
        network_links_changed = False
        node_ram_gb_changed = False
        for key, value in config_dict.items():
            if key not in vars(obj):
                raise KeyError(f"Unrecognized key {key!r}")

            match key:
                case "bandwidth_base_trace_path":
                    setattr(obj, key, Path(value))

                case "network":
                    network_changed = True
                    setattr(obj, key, value)

                case "network_links":
                    network_links_changed = True
                    setattr(obj, key, value)

                case "perfmodel_params_changed":
                    perfmodel_params_changed = True
                    setattr(obj, key, value)

                case "node_ram_gb":
                    node_ram_gb_changed = False
                    setattr(obj, key, value)

                case _:
                    setattr(obj, key, value)

        # If network has changed, also the nodes are changed. We need to make
        # sure also perfmodel_params and network_links have been updated,
        # otherwise they have just the default nodes/links.
        #
        # In case the updated params are not given, set to None and they will be
        # automatically generated with default values.
        if network_changed:
            if not perfmodel_params_changed:
                obj.perfmodel_params = None

            if not network_links_changed:
                obj.network_links = None

            if not node_ram_gb_changed:
                obj.node_ram_gb = None

        return obj


def _main():
    """Main entry point for dfaas_env_config script."""
    # Import these modules only if this module is called as main script.
    import argparse
    import yaml
    from datetime import datetime

    desc = "Write a DFaaSConfig configuration."
    parser = argparse.ArgumentParser(
        prog="dfaas_env_config", description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/env/default.yaml"), help="Override output file path"
    )

    args = parser.parse_args()

    config_dump = yaml.dump(DFaaSConfig().to_dict(), sort_keys=True, indent=4)
    now = datetime.now().astimezone().isoformat()
    class_filename = Path(__file__).name

    with args.output.open(mode="w") as out:
        print("# Automatically generated default DFaaSConfig configuration.", file=out)
        print(f"# See {class_filename!r} for more information.", file=out)
        print(f"# Generated on {now}", file=out)
        print("", file=out)
        out.write(config_dump)


if __name__ == "__main__":
    _main()
