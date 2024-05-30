# Multi-Agent RL for DFaaS by Emanuele Petriglia

This repository is in a working in progress state.

# Configuration setup

Minimal `requirements.txt` file:

```
ray[rrlib]==2.10.0
torch
gymnasium
dm_tree
pyarrow
pandas
typer
scikit-image
lz4
```

Only Ray (RRLib) must be bounded to a specific version (here 2.10.0).

Command to set up the environment:

```
$ dnf install python3.10
$ mkdir ray
$ cd ray
$ python3.10 -m venv .env
$ source .env/bin/activate
$ pip install --requirement requirements.txt
$ git clone https://github.com/FFede0/RL4CC.git
```

The RL4CC module must be installed under `ray/RL4CC` directory.
