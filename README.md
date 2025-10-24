# Basic Transformer For next character prediction

This repository contains a minimal character-level Transformer (decoder-only) implemented in JAX/Flax for next-character prediction. 

Repository structure
--------------------

Top-level layout:

- `transformer.ipynb` - Primary Jupyter notebook used for experimenting, training and generation. The notebook contains data loading, model initialization, training loop, and a JITted token generator cell.
- `models/` - Python package containing the Flax model implementation.
	- `models/models.py` - Minimal, decoder-only Transformer implementation (token & positional embeddings, DecoderBlocks, MLP, weight tying, causal attention).
- `data/` - a preprocessed `text8_dataset` used in the notebook.


Notes and pointers
------------------

- The notebook and model are intentionally small and pedagogical. They are a good starting point.
- The performance of the implemented model is (extremely) bad. There is a large room for experimentation and improvement.
