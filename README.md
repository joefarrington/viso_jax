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

## Running experiments

## Colab examples
