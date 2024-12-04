# Multi-Agent RL for DFaaS by Emanuele Petriglia

This repository is in a **working in progress** state.

If you are looking for the source code of the experiments of Emanuele
Petriglia's master's thesis, discussed in October 2024, see the
[`petriglia-thesis-2024`](https://github.com/unimib-datAI/marl-dfaas/tree/petriglia-thesis-2024)
branch.

The thesis, a summary and the presentation slides are available in another
[repository hosted on GitLab](https://gitlab.com/ema-pe/master-degree-thesis),
but they are written in Italian.

## How to set up the environment

The experiments are run and tested on Ubuntu 24.04 using Python 3.12. For a
reproducible development environment, it is preferable to install the
dependencies in a virtual environment (see the
[`venv`](https://docs.python.org/3.12/library/venv.html) module). The `venv`
module is not installed by default in Ubuntu, it must be installed using `sudo
apt install python3.12-venv`.

The complete list of Python dependencies can be found in the requirements.txt
file. However, the most important dependencies are:

* [Ray RLlib](https://docs.ray.io/en/releases-2.40.0/rllib/) (version 2.40):
  this is a reinforcement learning library used to define the DFaaS custom
  environment, run the experiments by training the models with the implemented
  algorithms.

* [PyTorch](https://pytorch.org/docs/2.5/) (version 2.5.1): PyTorch is a
  library for deep learning on GPUs and CPUs. It is used by Ray RLlib when
  training models with deep learning reinforcement learning algorithms.

* Gymnasium

* Matplotlib

* Flake8

When installing Ray RLlib, `pip` automatically installs its dependencies, which
are also used by the experiment scripts (like NumPy or Gymnasium). This means
that the environment can be easily set up by installing the following packages:

```
ray[rllib]==2.40.0
torch==2.5.1
gputil==1.4.0  # Required by RLlib (GPU system monitoring).
```

(OLD, WIP) Minimal `requirements.txt` file:

```
numpy==1.26.4
gputil
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

Run the following commands to set up the development environment with Ubuntu:

```
$ sudo apt install python3.12-venv
$ git clone https://github.com/unimib-datAI/marl-dfaas.git
$ cd marl-dfaas
$ python3.12 -m venv .env
$ source .env/bin/activate
$ pip install ray[rllib]==2.39.0 torch==2.5.1 gpuutil==1.4.0
```

For perfect reproducibility, there is a [`requirements.txt`](requirements.txt)
file that can be used instead of the previous command:

    $ pip install --requirement requirements.txt

Please note that both the requirements file and the command line suggestions
expect a machine with an NVIDIA GPU and CUDA (at least 12.4) installed for
PyTorch. PyTorch can also be used with a CPU, in this case follow the
[instructions](https://pytorch.org/get-started/locally/) on the official
website.

## How to run the experiments

WIP

## Patching Ray

The selected version of Ray RLlib needs to be patched to fix some bugs or
undesirable behaviour that has not yet been addressed upstream. The patches are
collected in the [`patches`](patches) directory and can be applied using the
[`patch`](https://www.man7.org/linux/man-pages/man1/patch.1.html) command:

    patch -p0 < patches/NAME.patch

The patches have only been tested with Ray 2.40.0. They will only work if the
virtual environment is named `.env` and the Python version is 3.12, as the file
path is hardcoded into the patch file.

Note: The `patch` binary is required and preinstalled on Ubuntu. If not
available, it can be installed with `apt install patch`.

The patches are created using the standard
[`diff`](https://www.man7.org/linux/man-pages/man1/diff.1.html) tool:

    diff -Naru .env/.../rllib/example.py .env/.../rllib/example_new.py > patches/NAME.patch

See [this reply](https://unix.stackexchange.com/a/162146) on StackExchange for
more information.

## License

Copyright 2024 Emanuele Petriglia

The source code in this repository is licensed under the Apache License,
version 2.0. See the [LICENSE](LICENSE) file for more information.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.
