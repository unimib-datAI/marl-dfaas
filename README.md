# Multi-Agent RL for DFaaS by Emanuele Petriglia

This repository is in a working in progress state.

# Configuration setup

Minimal `requirements.txt` file:

```
numpy==1.26.4
ray[rrlib]==2.10.0
torch
gymnasium
dm_tree
pyarrow
pandas
typer
scikit-image
lz4
flake8
matplotlib
```

Ray (specifically RRLib) must be bound to version 2.10.0, otherwise the code
(and RL4CC) may not work properly. Also, NumPy 2.x is not yet supported, only
the latest 1.x version.

To set up the development environment with Fedora:

```
$ dnf install python3.10
$ mkdir ray
$ cd ray
$ python3.10 -m venv .env
$ source .env/bin/activate
$ pip install --requirement requirements.txt
$ git clone https://github.com/FFede0/RL4CC.git
```

The RL4CC module must be installed under `ray/RL4CC` directory. This git
repository must be on the test branch, it has been tested with the b9146c1
commit and some additional custom patches.

The `single-agent` directory contains the code for the single agent version of
DFaaS workload distribution using reinforcement learning, loosely based on
Giacomo Pracucci's thesis and code.

Run `make clean` to clean the `results` directory, which stores the experiment
logs, metrics, results, and plots for each experiment run.

## Patching Ray and RL4CC

`patch` binary is required. WIP
