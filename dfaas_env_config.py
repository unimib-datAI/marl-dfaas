"""Configuration module for the DFaaS multi-agent environment.

This module provided DFaaSConfig, a class that follows the build pattern to
create, validate and enrich the configuration for the DFaaS environment.

I took inspiration from the PPOConfig class defined in Ray RLLib (2.X old
stack).
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
    """Defines a configuration class from which a DFaaS env can be built."""

    def __init__(self):
        """Initializes a DFaaSConfig instance with default values."""

        # Network structure given as Networkx's adjacency list.
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

        # RAM capacity in GB for each node.
        self.node_ram_gb = {"node_0": 4, "node_1": 4}

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
        # Note: if maximum_concurrency is None, the value will be calculated
        # based on request_memory_mb and node_ram_gb.
        self._perfmodel_params = {
            "warm_service_time": 1,
            "cold_service_time": 2,
            "idle_time_before_kill": 30,
            "maximum_concurrency": None,
        }

        # Performance model parameters for each node in the network.
        self.perfmodel_params = {"node_0": self._perfmodel_params.copy(), "node_1": self._perfmodel_params.copy()}

        # Seed used when building the environment. Warning: it is not used
        # to reset the environment!
        self.build_seed = 42

    def validate(self):
        """Validates configuration. Automatically called by build().

        Raise ValueError, TypeError or KeyError on validation error."""
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

        # self.request_memory_mb validation.
        if self.request_memory_mb is None:
            raise TypeError("request_memory_mb should not be None")

        # Validate perfmodel_params for all nodes.
        perfmodel_keys = set(self._perfmodel_params.keys())
        for node in self._network.nodes:
            # Node exists?
            if node not in self.perfmodel_params:
                raise ValueError(f"perfmodel_params: node {node!r} missing")

            # Node has all keys?
            node_perfmodel_keys = self.perfmodel_params[node].keys()
            if perfmodel_keys != node_perfmodel_keys:
                raise ValueError(f"perfmodel_params: node {node!r} has different set of keys!")

            # All required keys have the right values?
            warm_sv_time = self.perfmodel_params[node]["warm_service_time"]
            cold_sv_time = self.perfmodel_params[node]["cold_service_time"]
            idle_time = self.perfmodel_params[node]["idle_time_before_kill"]
            max_conc = self.perfmodel_params[node]["maximum_concurrency"]
            if warm_sv_time <= 0:
                raise ValueError(
                    f"perfmodel_params: node {node!r}: warm_service_time must be positive, got {warm_sv_time}"
                )
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
        for node in self._network.nodes:
            # Node exists?
            if node not in self.node_ram_gb:
                raise ValueError(f"node_ram_gb: node {node!r} missing")

            # Node has the right value?
            if self.node_ram_gb[node] <= 0:
                raise ValueError(f"node_ram_gb: node {node!r}: must be positive, got {self.node_ram_gb[node]}")

        # Validate network_links for all edges.
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
            if link_params["bandwidth_mbps"] is None:
                raise ValueError(f"network_links: undirected link ({u}, {v}): bandwidth_mbps should not be None")
            if len(link_params["bandwidth_mbps"]) != self.max_steps:
                raise ValueError(f"network_links: undirected link ({u}, {v}): bandwidth_mbps trace has wrong length")

            seen.add((u, v), (v, u))

    def build(self):
        """Build the DFaaS environment from this configuration.

        Return an initialized (but not started) DFaaS environment."""
        rng = np.random.default_rng(seed=self.build_seed)

        # Generate NetworkX graph (internal attribute).
        self._network = nx.parse_adjlist(self.network)

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

        # Calculate maximum_concurrency for each node.
        for node in self._network.nodes:
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

        return DFaaS(config=self)

    def to_dict(self):
        """Convert DFaaSConfig to a dict, safe to serialize as JSON or YAML."""
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
        for key, value in config_dict:
            if key not in vars(obj):
                raise KeyError(f"Unrecognized key {key!r}")

            match key:
                case "bandwidth_base_trace_path":
                    setattr(obj, key, Path(value))

                case _:
                    setattr(obj, key, value)

        return obj
