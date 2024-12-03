# Multi-Agent RL for DFaaS by Emanuele Petriglia

This repository is in a working in progress state.

# Configuration setup

Minimal `requirements.txt` file:

```
numpy==1.26.4
gputil
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
$ mkdir dfaas-rl-petriglia
$ cd dfaas-rl-petriglia
$ python3.10 -m venv .env
$ source .env/bin/activate
$ pip install --requirement requirements.txt
```

## RL4CC

The RL4CC module is optional and is only required for single agent experiments.
It must be installed in the `dfaas-rl-petriglia/RL4CC` directory:

    $ git clone https://github.com/FFede0/RL4CC.git

This git repository must be on the test branch, it has been tested with the
b9146c1 commit and some additional custom patches.

The `single-agent` directory contains the code for the single agent version of
DFaaS workload distribution using reinforcement learning, loosely based on
Giacomo Pracucci's thesis and code.

## Patching Ray

The selected Ray version needs to be patched to fix some unwanted behaviour. The
patches are collected in the `patches` directory and can be applied using the
[`patch`](https://www.man7.org/linux/man-pages/man1/patch.1.html) command:

    patch -p0 < patches/NAME.patch

The patches have only been tested with Ray 2.10. They only work if the virtual
environment is named `.env` and the Python version is 3.10.

Note: the `patch` binary is required and is not installed by default in Fedora,
it can be installed with `dnf install patch`.

The patches are created using the standard
[`diff`](https://www.man7.org/linux/man-pages/man1/diff.1.html) tool:

    diff -Naru .env/.../rllib/example.py .env/.../rllib/example_new.py > patches/NAME.patch

See [this reply](https://unix.stackexchange.com/a/162146) on StackExchange for more information.

## License

Copyright 2024 Emanuele Petriglia

The source code in this repository is licensed under the Apache License,
version 2.0. See the [LICENSE](LICENSE) file for more information.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.
