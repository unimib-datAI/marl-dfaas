"""Configuration module for the DFaaS multi-agent environment.

This modules provides the configuration as dataclasses.

Create DFaaSConfig() to get the default configuration.
"""

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats

from bandwidth_generator import generate_traces


@dataclass
class NetworkLinkConfig:
    """Configuration for a network link between two nodes.

    The bandwidth trace can be given in three ways:

    1. Use "generated" method, and it will be automatically generated around the
    given mean and with the given random noise.
    2. Use "static" method, and the given mean value will be repeated for all
    steps.
    3. Use "static" method and give directly the bandwidth trace in
    bandwidth_mbps.
    """

    # Access delay in milliseconds (for latency).
    access_delay_ms: float = 5.0

    # Method for bandwidth trace generation.
    bandwidth_mbps_method: Literal["static", "generated"] = "generated"

    # Mean bandwidth value in Mbps, used for generation.
    bandwidth_mbps_mean: float = 100.0

    # Random noise factor for bandwidth trace generation [0, 1]. Useful only if
    # bandwidth_mbps_method="generated".
    bandwidth_mbps_random_noise: float = 0.1

    # Actual bandwidth trace, a list of max_steps values. Usually this field is
    # auto-generated based on the other fields.
    bandwidth_mbps: List[float] = None

    def validate(self, max_steps: int):
        """Validate configuration.

        Args:
            max_steps: Number of steps in the episode.

        Raises:
            ValueError: If there is a validation error.
        """
        if self.access_delay_ms < 0:
            raise ValueError(f"access_delay_ms must be non-negative, got {self.access_delay_ms}")

        if self.bandwidth_mbps_mean <= 0:
            raise ValueError(f"bandwidth_mbps_mean must be positive, got {self.bandwidth_mbps_mean}")

        if not 0.0 <= self.bandwidth_mbps_random_noise <= 1.0:
            raise ValueError(f"bandwidth_mbps_random_noise must be in [0, 1], got {self.bandwidth_mbps_random_noise}")

        if self.bandwidth_mbps_method not in ["static", "generated"]:
            raise ValueError(
                f"bandwidth_mbps_method must be 'static' or 'generated', got {self.bandwidth_mbps_method!r}"
            )

        if not self.bandwidth_mbps:
            raise ValueError("bandwidth_mbps trace has not been generated!")

        if len(self.bandwidth_mbps) != max_steps:
            raise ValueError(f"bandwidth_mbps trace has wrong length {len(self.bandwidth_mbps)} != {max_steps}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NetworkLinkConfig":
        """Create NetworkLinkConfig from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PerfModelParams:
    """Performance model parameters for a single agent/node.

    Attributes:
        warm_service_time: Service time for warm starts (seconds)
        cold_service_time: Service time for cold starts (seconds)
        idle_time_before_kill: Time before idle container is killed (seconds)
        maximum_concurrency: Maximum number of concurrent requests the node can handle
    """

    # Service time for wam starts in seconds.
    warm_service_time: float

    # Cold service time in seconds.
    cold_service_time: float

    # Time before idle container is killed in seconds.
    idle_time_before_kill: float

    # Maximum number of concurrent requests a node can handle. Will be
    # calculated if not provided.
    maximum_concurrency: Optional[float] = 0.0

    def __post_init__(self):
        """Validate performance parameters."""
        if self.warm_service_time < 0:
            raise ValueError(f"warm_service_time must be non-negative, got {self.warm_service_time}")

        if self.cold_service_time < 0:
            raise ValueError(f"cold_service_time must be non-negative, got {self.cold_service_time}")

        if self.cold_service_time < self.warm_service_time:
            raise ValueError(
                f"cold_service_time ({self.cold_service_time}) should be >= "
                f"warm_service_time ({self.warm_service_time})"
            )

        if self.idle_time_before_kill < 0:
            raise ValueError(f"idle_time_before_kill must be non-negative, got {self.idle_time_before_kill}")

        if self.maximum_concurrency < 0:
            raise ValueError(f"maximum_concurrency must be non-negative, got {self.maximum_concurrency}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerfModelParams":
        """Create PerfModelParams from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DFaaSConfig:
    """Main configuration for the DFaaS multi-agent environment.

    This dataclass encapsulates all configuration parameters for the DFaaS
    environment.
    """

    # Network structure given as Networkx's adjacency list.
    network: List[str] = field(default_factory=lambda: ["node_0 node_1"])

    # Network link parameters for each link in the network.
    network_links: Dict[str, Dict[str, NetworkLinkConfig]] = field(default_factory=dict)

    # Path to the bandwidth 5G base trace, used to generate the bandwidth
    # traces. This field is used only if at least one link uses the "generated"
    # bandwidth generation method.
    bandwidth_base_trace_path: Optional[Path] = None

    # Number of steps for an episode.
    max_steps: int = 288

    # Input rate generation configuration.
    input_rate_method: str = "synthetic-sinusoidal"
    input_rate_same_method: bool = True

    # Environment mode.
    evaluation: bool = False

    # Performance model parameters.
    perfmodel_params: Dict[str, PerfModelParams] = field(default_factory=dict)

    # Data input size (in bytes) of the single function. The size values is
    # extracted from a truncated normal distribution with the given range
    # (inclusive) and given mean and standard deviation.
    #
    # To have a single value, just put the range [min, min] and mean/std to
    # [min, 0].
    request_input_data_size_bytes_range: List[int] = field(default_factory=lambda: [100, 5242880])
    request_input_data_size_bytes_mean_std: List[float] = field(default_factory=lambda: [1024.0, 1024.0])

    # Memory demand (in MB) for the single function. The demand value is
    # extracted uniformly from the given (inclusive) range.
    #
    # To have a single value, just put the range to [min, min].
    request_memory_mb_range: List[int] = field(default_factory=lambda: [128, 1024])

    # RAM capacity in GB for each node.
    node_ram_gb: Dict[str, float] = field(default_factory=dict)

    # Generated values.
    request_input_data_size_bytes: Optional[int] = None
    request_memory_mb: Optional[int] = None

    # Seed for random generation (optional).
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization and auto-generate values."""
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        if len(self.request_input_data_size_bytes_range) != 2:
            raise ValueError(
                f"request_input_data_size_bytes_range must be [min, max], "
                f"got {len(self.request_input_data_size_bytes_range)} values"
            )

        if len(self.request_input_data_size_bytes_mean_std) != 2:
            raise ValueError(
                f"request_input_data_size_bytes_mean_std must be [mean, std], "
                f"got {len(self.request_input_data_size_bytes_mean_std)} values"
            )

        if len(self.request_memory_mb_range) != 2:
            raise ValueError(
                f"request_memory_mb_range must be [min, max], got {len(self.request_memory_mb_range)} values"
            )

        # Validate ranges.
        if self.request_input_data_size_bytes_range[0] > self.request_input_data_size_bytes_range[1]:
            raise ValueError("request_input_data_size_bytes_range: min > max")

        if self.request_memory_mb_range[0] > self.request_memory_mb_range[1]:
            raise ValueError("request_memory_mb_range: min > max")

        # Convert network_links dict values to NetworkLinkConfig if they're
        # plain dicts.
        for src in self.network_links:
            for dest in self.network_links[src]:
                if isinstance(self.network_links[src][dest], dict):
                    self.network_links[src][dest] = NetworkLinkConfig.from_dict(self.network_links[src][dest])

        # Convert perfmodel_params dict values to PerfModelParams if they're
        # plain dicts.
        for agent in list(self.perfmodel_params.keys()):
            if isinstance(self.perfmodel_params[agent], dict):
                self.perfmodel_params[agent] = PerfModelParams.from_dict(self.perfmodel_params[agent])

        # Auto-generate all values with the provided seed (or random if None)
        rng = np.random.default_rng(seed=self.seed)
        self.generate_all(rng)

    def generate_request_input_data_size(self, rng: np.random.Generator) -> None:
        """Generate request input data size from truncated normal distribution.

        The value is sampled from a truncated normal distribution within a range,
        a mean and a standard deviation.

        Default values extracted from the original article, DOI:
        10.1016/j.future.2024.02.019

        Args:
            rng: Random number generator
        """
        if self.request_input_data_size_bytes is not None:
            return  # Already generated.

        mean, std = self.request_input_data_size_bytes_mean_std
        min_val, max_val = self.request_input_data_size_bytes_range

        # Normalized bounds for truncnorm.
        #
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        a = (min_val - mean) / std
        b = (max_val - mean) / std

        self.request_input_data_size_bytes = int(scipy.stats.truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))

    def generate_request_memory(self, rng: np.random.Generator) -> None:
        """Generate request memory demand uniformly from configured range.

        Memory demand (in MB) for a request.


        Args:
            rng: Random number generator
        """
        if self.request_memory_mb is not None:
            return  # Already generated

        min_mem, max_mem = self.request_memory_mb_range
        self.request_memory_mb = int(rng.integers(min_mem, high=max_mem, endpoint=True))

    def calculate_maximum_concurrency(self) -> None:
        """Calculate maximum concurrency for each agent based on RAM and memory demand.

        Maximum concurrency.

        Required values: request_memory_mb and node_ram_gb.

        Requires:
            - request_memory_mb must be set
            - node_ram_gb must be populated for all agents
            - perfmodel_params must be initialized for all agents

        Raises:
            ValueError: If required values are not set
        """
        if self.request_memory_mb is None:
            raise ValueError("request_memory_mb must be generated before calculating concurrency")

        agents = self.get_agents()

        for agent in agents:
            if agent not in self.node_ram_gb:
                raise ValueError(f"node_ram_gb not set for agent {agent}")
            if agent not in self.perfmodel_params:
                raise ValueError(f"perfmodel_params not set for agent {agent}")

            max_concurrency = float(np.floor((self.node_ram_gb[agent] * 1024) / self.request_memory_mb))
            self.perfmodel_params[agent].maximum_concurrency = max_concurrency

    def generate_bandwidth_trace(self, rng: np.random.Generator) -> None:
        """Generate bandwidth trace, based on the configured method defined for
        each link.

        Args:
            rng: Random number generator
        """
        # Load base trace if provided.
        if self.bandwidth_base_trace_path:
            df = pd.read_csv(self.bandwidth_base_trace_path)
            base_trace = df["Throughput"].to_numpy()
        else:
            base_trace = None

        # Count how many links want to generate the bandwidth trace.
        generated = 0
        random_noise, target_mean = None, None
        for src in self.network_links:
            for dest in self.network_links[src]:
                params = self.network_links[src][dest]

                if params.bandwidth_mbps_method == "generated":
                    generated += 1

                    # FIXME: We currently support only one random noise factor
                    # value, that must be the same for all links. So we need to
                    # check manually all values to be the same.
                    #
                    # The same also for the mean.
                    if not random_noise:
                        random_noise = params.bandwidth_random_noise
                    elif random_noise != params.bandwidth_random_noise:
                        raise ValueError(
                            "bandwidth_mbps_random_noise must be the same value for all links (known limitation)"
                        )
                    if not target_mean:
                        target_mean = params.bandwidth_mbps_mean
                    elif target_mean != params.bandwidth_mbps_mean:
                        raise ValueError("bandwidth_mbps_mean must be the same value for all links (known limitation)")

                elif not params.bandwidth_mbps:
                    # This link has set the "static" method and provided just a
                    # single value: expand to a static trace.
                    params.bandwidth_mbps = np.full(self.max_steps, fill_value=params.bandwidth_mbps_mean).tolist()

                else:
                    # This link has set manually a complete trace. Use as is.
                    pass

        if generated > 0:
            if not base_trace:
                raise ValueError("bandwidth_base_trace_path needed but not given")

            # Generate the traces.
            traces = generate_traces(
                base_trace=base_trace,
                num_traces=generated,
                max_len=self.max_steps,
                random_noise=random_noise,
                seed=self.seed,
                target_mean=target_mean,
            )

            i = 0
            # Now assign a trace to each link.
            for src in self.network_links:
                for dest in self.network_links[src]:
                    params = self.network_links[src][dest]

                    if params.bandwidth_mbps_method == "generated":
                        params.bandwidth_mbps = traces[i]
                        i += 1

    def initialize_defaults(self) -> None:
        """Initialize default values for node resources, performance params, and network links.

        Sets:
            - Default node RAM (4 GB per agent) if not provided
            - Default performance model parameters if not provided
            - Default network link parameters for all edges if not provided
        """
        agents = self.get_agents()
        network = nx.parse_adjlist(self.network)

        # RAM (in GB) per each node.
        for agent in agents:
            if agent not in self.node_ram_gb:
                self.node_ram_gb[agent] = 4.0

        # Initialize perfmodel_params if not present in the config.
        if not self.perfmodel_params:
            default_params = PerfModelParams(
                warm_service_time=0.5,
                cold_service_time=1.25,
                idle_time_before_kill=60.0,
                maximum_concurrency=0.0,  # Will be calculated later
            )
            self.perfmodel_params = {agent: default_params for agent in agents}
        else:
            # Ensure all agents have perfmodel_params
            default_params = PerfModelParams(
                warm_service_time=0.5, cold_service_time=1.25, idle_time_before_kill=60.0, maximum_concurrency=0.0
            )
            for agent in agents:
                if agent not in self.perfmodel_params:
                    self.perfmodel_params[agent] = default_params

        # Default values for the network link parameters.
        for u, v in network.edges():
            self.network_links.setdefault(u, {})
            if v not in self.network_links[u]:
                self.network_links[u][v] = NetworkLinkConfig()

    def generate_all(self, rng: np.random.Generator) -> None:
        """Generate all configuration values in the correct order.

        This method orchestrates the generation of all derived and random values:
        1. Initialize defaults for nodes and links
        2. Generate request characteristics (data size and memory)
        3. Calculate maximum concurrency based on resources
        4. Generate bandwidth traces for all network links
        5. Validate the complete configuration

        Args:
            rng: Random number generator
        """
        # Step 1: Initialize defaults
        self.initialize_defaults()

        # Step 2: Generate request characteristics
        self.generate_request_input_data_size(rng)
        self.generate_request_memory(rng)

        # Step 3: Calculate derived values
        self.calculate_maximum_concurrency()

        # Step 4: Generate bandwidth traces for all network links
        self.generate_bandwidth_trace(rng)

        # Step 5: Validate everything
        self.validate()

    def validate(self) -> None:
        """Validate the complete configuration for consistency.

        Raises:
            ValueError: If configuration is invalid or inconsistent
        """

        # Parse network to get agents.
        network = nx.parse_adjlist(self.network)
        agents = list(network.nodes)

        # Validate perfmodel_params has entries for all agents.
        for agent in agents:
            if agent not in self.perfmodel_params:
                raise ValueError(f"Agent {agent!r} missing in perfmodel_params")

        # Validate network_links for all edges.
        for u, v in network.edges():
            if u not in self.network_links:
                raise ValueError(f"Source node {u!r} missing in network_links")
            if v not in self.network_links[u]:
                raise ValueError(f"Link ({u}, {v}) missing in network_links")

            self.network_links[u][v].validate(max_steps=self.max_steps)

        # Validate node_ram_gb.
        for agent in agents:
            if agent in self.node_ram_gb:
                if self.node_ram_gb[agent] <= 0:
                    raise ValueError(f"Node RAM for {agent!r} must be positive")

        # Validate generated values if present.
        if self.request_input_data_size_bytes is not None:
            min_size, max_size = self.request_input_data_size_bytes_range
            if not (min_size <= self.request_input_data_size_bytes <= max_size):
                raise ValueError(
                    f"request_input_data_size_bytes ({self.request_input_data_size_bytes}) "
                    f"outside range [{min_size}, {max_size}]"
                )

        if self.request_memory_mb is not None:
            min_mem, max_mem = self.request_memory_mb_range
            if not (min_mem <= self.request_memory_mb <= max_mem):
                raise ValueError(f"request_memory_mb ({self.request_memory_mb}) outside range [{min_mem}, {max_mem}]")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dictionary representation compatible with original dict-based config
        """
        result = {}

        # Simple fields.
        result["network"] = self.network
        result["max_steps"] = self.max_steps
        result["input_rate_method"] = self.input_rate_method
        result["input_rate_same_method"] = self.input_rate_same_method
        result["evaluation"] = self.evaluation

        # Convert network_links.
        result["network_links"] = {}
        for src, dests in self.network_links.items():
            result["network_links"][src] = {}
            for dest, config in dests.items():
                result["network_links"][src][dest] = config.to_dict()

        # Convert perfmodel_params.
        result["perfmodel_params"] = {}
        for agent, params in self.perfmodel_params.items():
            result["perfmodel_params"][agent] = params.to_dict()

        # Request configuration.
        result["request_input_data_size_bytes_range"] = self.request_input_data_size_bytes_range
        result["request_input_data_size_bytes_mean_std"] = self.request_input_data_size_bytes_mean_std
        result["request_memory_mb_range"] = self.request_memory_mb_range

        # Node resources.
        if self.node_ram_gb:
            result["node_ram_gb"] = self.node_ram_gb

        # Generated values.
        if self.request_input_data_size_bytes is not None:
            result["request_input_data_size_bytes"] = self.request_input_data_size_bytes

        if self.request_memory_mb is not None:
            result["request_memory_mb"] = self.request_memory_mb

        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DFaaSConfig":
        """Create DFaaSConfig from dictionary.

        Args:
            d: Dictionary with configuration values

        Returns:
            DFaaSConfig instance
        """
        # Extract fields that match the dataclass.
        kwargs = {}

        for field_name in cls.__dataclass_fields__:
            if field_name in d:
                kwargs[field_name] = d[field_name]

        return cls(**kwargs)

    def get_agents(self) -> List[str]:
        """Get list of agent IDs from the network configuration."""
        network = nx.parse_adjlist(self.network)
        return list(network.nodes)
