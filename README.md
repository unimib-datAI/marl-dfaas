# Multi-Agent RL for DFaaS by Emanuele Petriglia

This repository is in a **working in progress** state.

If you are looking for the source code of the experiments of Emanuele
Petriglia's master's thesis, discussed in October 2024, see the
[`petriglia-thesis-2024`](https://github.com/unimib-datAI/marl-dfaas/tree/petriglia-thesis-2024)
branch.

The thesis, a summary and the presentation slides are available in another
[repository hosted on GitLab](https://gitlab.com/ema-pe/master-degree-thesis),
but they are written in Italian.

## Project directory structure

* `dataset`: contains the dataset used to have real function invocations traces
  in the DFaaS environment (instead of generated ones).
* `models`: the neural network models used for PPO and SAC algorithms, specified
  as JSON files.
* `notebooks`: Python Jupyter notebooks used mainly to show plots of the
  experiments and do some simple prototyping or experiments.
* `patches`: required custom patches to Ray RLlib or other libraries needed to
  run the experiments.
* `plots`: non-interactive Python scripts to generate plots from experiments.
* `results`: default directory where the experiment data is stored. This folder
  is not shown in the repository because it contains ephemeral data.
* `tests`: some Python scripts used to test the Ray RLLib, the DFaaS
  environment, or other miscellaneous tests.

## How to set up the environment

The experiments are run and tested on Ubuntu 24.04 using Python 3.12. For a
reproducible development environment, it is preferable to install the
dependencies in a virtual environment (see the
[`venv`](https://docs.python.org/3.12/library/venv.html) module). The `venv`
module is not installed by default in Ubuntu, it must be installed using `sudo
apt install python3.12-venv`.

The complete list of Python dependencies can be found in the requirements.txt
and requirements.base.txt files. However, the most important dependencies are:

* [Ray RLlib](https://docs.ray.io/en/releases-2.40.0/rllib/):
  this is a reinforcement learning library used to define the DFaaS custom
  environment, run the experiments by training the models with the implemented
  algorithms. The version is pinned to 2.40.0.

* [PyTorch](https://pytorch.org/docs/2.5/): is a library for deep learning on
  GPUs and CPUs. It is used by Ray RLlib when training models with deep learning
  reinforcement learning algorithms.

For plotting only, the following dependencies are required:

* [Matplotlib](https://matplotlib.org/): is a plot generation library used in
  the scripts in the [`plots`](plots) and [`notebooks`](notebooks) directory.

* [orjson](https://pypi.org/project/orjson/): is a JSON library that is faster
  than the standard library. Used because the experiments generate large JSON
  files that slow down the encoding/decoding processes.

* [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/v7.3.2/):
  Jupyter notebooks are used to create plots and explore results and statistics
  from the experiments. The notebooks are stored in the `notebooks` directory. 

* [ipympl](https://matplotlib.org/ipympl/): this is an extension for Jupyter
  Notebook to support interactive Matplotlib using Jupyter Widgets
  [`ipywidgets`](https://ipywidgets.readthedocs.io/en/latest/index.html#).

* [nbstripout](https://pypi.org/project/nbstripout/): utility that erases a
  Jupyter Notebook's output before committing to git.

Only the first two dependencies (Ray RLlib and PyTorch) are required. The other
dependencies are required only if you are using the notebooks. Scripts in the
`plots` directory are not interactive and only require Matplotlib.

When installing Ray RLlib, `pip` automatically installs some dependencies used
by the project, like NumPy, Pandas or Gymnasium. This means that the environment
can be easily set up by installing the dependencies listed above and they can be
found also in the [`requirements.base.txt`](requirements.base.txt).

Run the following commands to set up the development environment with Ubuntu:

```
$ sudo apt install python3.12-venv
$ git clone https://github.com/unimib-datAI/marl-dfaas.git
$ cd marl-dfaas
$ python3.12 -m venv .env
$ source .env/bin/activate
$ pip install --requirement requirements-base.txt
```

For perfect reproducibility, there is a [`requirements.txt`](requirements.txt)
that contains all the dependencies installed with the fixed versions:

    $ pip install --requirement requirements.txt

Please note that the requirements file expects a machine with an NVIDIA GPU and
CUDA (at least 12.4) installed for PyTorch. PyTorch can also be used with a CPU,
in this case follow the [instructions](https://pytorch.org/get-started/locally/)
on the official website.

The requirements file also contains [`black`](https://black.readthedocs.io) (a
development tool for automatically formatting source code and Jupyter
notebooks), [`pylint`](https://pylint.readthedocs.io/en/latest/index.html) (a
static code analyser) and [`pre-commit`](https://pre-commit.com) packages. The
latter run automatically `black` when doing commits.

## How to run the experiments

WIP

**Important**: always run Python scripts from the project root directory to
allow loading of commonly used modules (`dfaas_env.py`...). As example, if you
need to run a test script:

    $ python tests/env/local_strategy.py

### How to run Jupyter notebooks

Just run:

    $ jupyter notebook --no-browser notebooks/

Then open the link in the output in a browser.

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

Copyright © 2024-2025 Emanuele Petriglia

The source code in this repository is licensed under the Apache License,
version 2.0. See the [LICENSE](LICENSE) file for more information.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.
