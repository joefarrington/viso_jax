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
