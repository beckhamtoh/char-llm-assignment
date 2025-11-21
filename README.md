# Basic Transformer For next character prediction

This repository contains a minimal character-level Transformer (decoder-only) implemented in JAX/Flax for next-character prediction.

## Repository structure

# Character-Level Language Model Research

## Project Structure

### data

- `text8_test.txt` - Test dataset
- `text8_train.txt` - Training dataset

### models

- `models.py` - Core model architecture definitions

### util

- `util/generation.py` - Autoregressive text generation with temperature sampling

### Notebooks

- `metrics_and_loss.ipynb` - Loss tracking and metrics analysis
- `model_tuning_positional.ipynb` - Positional encoding comparison experiments
- `model_tuning_scaling_law.ipynb` - Scaling laws analysis and fitting
- `optimization_scaled.ipynb` - Optimization hyperparameter experiments using equal-compute budget scaling law
- `optimization_small_expts.ipynb` - Optimization hyperparameter experiments using small runs
- `predicting_multi_char.ipynb` - Multi-character prediction experiments
- `transformer.ipynb` - Initial transformer configuration (given)
- `transformer_final.ipynb` - Our final transformer configuration
