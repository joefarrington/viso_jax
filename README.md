# viso_jax

GPU-accelerated value iteration and simulation optimization for perishable inventory control using JAX

## Introduction

## Installation

To use JAX with Nvidia GPU-acceleration, you must first install CUDA and CuDNN. See the [JAX installation instructions](https://github.com/google/jax#installation) for further details.

Python dependencies are listed in `pyproject.toml`. We use [poetry](https://python-poetry.org/docs/) for dependency management.

viso_jax and its Python dependencies can be installed using the code snippet below. This snippet assumes that you have [poetry installed](https://python-poetry.org/docs/#installation). If this snippet not run in a virtual environment, poetry will create a new virtual environment before installing the dependencies.

```bash
git clone https://github.com/joefarrington/viso_jax.git
cd viso_jax
poetry install viso_jax
```

Once installation is complete, you can test that JAX recognises an accelerator (GPU or TPU) by running the following snippet:

```bash
poetry run pytest -m "jax"
```

## Scenarios

### de_moor_perishable

Adapted from <i>"Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management"</i> by [De Moor et al (2022)](https://doi.org/10.1016/j.ejor.2021.10.045)

### hendrix_perishable_one_product

Adapted from the single product scenarios in <i>"On computing optimal policies in perishable inventory control using value iteration"</i> by [Hendrix et al (2019)](https://doi.org/10.1002/cmm4.1027)

### hendrix_perishable_substitution_two_product

Adapted from the two product scenarios in <i>"On computing optimal policies in perishable inventory control using value iteration"</i> by [Hendrix et al (2019)](https://doi.org/10.1002/cmm4.1027)

Additional experimental settings are taken from [Ortega et al (2019)](https://doi.org/10.1007/s11227-018-2692-z).

### mirjalili_perishable_platelet

Adapted from the scenario in Chapter 6 of <i>"Data-driven modelling and control of hospital blood inventory"</i> by [Mirjalili (2022)](https://tspace.library.utoronto.ca/bitstream/1807/124976/1/Mirjalili_Mahdi_202211_PhD_thesis.pdf)

## Running experiments

## Colab examples

## Recommended resources

### JAX

#### Getting started

The [JAX documentation](https://jax.readthedocs.io/en/latest/index.html) includes [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html), a set of interactive introductory tutorials. We also recommed reading [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) to understand key differences between NumPy and JAX.

The [Awesome JAX](https://github.com/n2cholas/awesome-jax) GitHub repository contains links to a wide variety of Python libraries and projects based on JAX.

#### Value iteration

Thomas J Sargent and John Stachurski provide an [interactive tutorial](https://notes.quantecon.org/submission/622ed4daf57192000f918c61) implementing value iteration in JAX for an economics problem and comparing the speed of two NumPy-based approaches with GPU-accelerated JAX.

### Hydra

We specified the configurations of our experiments using [Hydra](https://hydra.cc/), which support composable configuration files and provides a command line interface for overriding configuration items.

### Gymnax

We created reinforcement learning environments for each scenario using [Gymnax](https://github.com/RobertTLange/gymnax). Gymnax provides an API similar to the OpenAI gym API, but allows simulated rollouts to run in parallel on GPU. This is particularly helpful for simulation optimization because it allows many possible parameters for the heuristic policies (e.g. a base-stock policy) to be evaluated at the same time, each on many parallel rollouts.

### Optuna

We used [Optuna](https://optuna.readthedocs.io/en/stable/) to search the parameter spaces for heuristic policies in our simulation optimization experiments.
