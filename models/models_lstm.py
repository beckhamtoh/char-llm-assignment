# models_lstm.py
import jax
import jax.numpy as jnp
import flax.linen as nn


class LSTMStack(nn.Module):
    """
    A simple stacked LSTM implemented with flax.linen LSTMCell and jax.lax.scan.

    Args:
      hidden_size: int, the number of features in each LSTM layer (H)
      n_layers:    int, number of stacked LSTM layers
      dropout:     float, dropout rate applied to each layer's output (0.0 disables)

    Input:
      x: (B, T, D) float32 — sequence of embeddings

    Returns:
      y: (B, T, H) float32 — sequence of hidden states from the top LSTM layer
      final_state: tuple of length n_layers, each element is an LSTMState(c, h)
    """
    hidden_size: int
    n_layers: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, init_state=None, deterministic: bool = True):
        B, T, _ = x.shape
        H = self.hidden_size

        # Build per-layer LSTM cells (features=H)
        cells = tuple(nn.LSTMCell(features=H, name=f"lstm_{i}") for i in range(self.n_layers))

        # Initialize carry for each layer if not provided (API variant without size kwarg)
        if init_state is None:
            states = tuple(
                cell.initialize_carry(jax.random.PRNGKey(0), (B,))
                for cell in cells
            )
        else:
            # Expect a tuple of nn.LSTMCell.LSTMState for each layer
            states = init_state

        def step(carry, x_t):
            """One time step through all layers."""
            layer_states = carry
            inp = x_t
            new_states = []
            for i, cell in enumerate(cells):
                state_i, out = cell(layer_states[i], inp)   # out: (B, H)
                # Residual-style dropout on layer outputs (simple, effective)
                out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
                new_states.append(state_i)
                inp = out
            return tuple(new_states), inp  # carry (states), output (top layer)

        # Scan across time: time-major in, batch-major out
        final_states, y_time_major = jax.lax.scan(step, states, jnp.swapaxes(x, 0, 1))
        y = jnp.swapaxes(y_time_major, 0, 1)  # (B, T, H)
        return y, final_states


class LSTMCharLM(nn.Module):
    """
    Character-level language model:
      token embedding -> stacked LSTM -> LayerNorm -> projection to vocab.

    - Weight tying (default): reuse the embedding matrix E to project to logits
      via einsum: logits[b,t,v] = dot(y[b,t,:], E[v,:]).

    Args:
      vocab_size: int, number of characters (V)
      d_model:    int, embedding size and LSTM hidden size (use same for simplicity)
      n_layers:   int, number of stacked LSTM layers
      tie_weights: bool, enable/disable weight tying (default True)
      dropout:    float, dropout rate inside LSTM stack outputs

    __call__(idx, init_state=None, deterministic=True):
      idx: (B, T) int32 — token ids in [0, V)
      Returns:
        logits: (B, T, V) float32 — unnormalized scores
        final_state: tuple of LSTM states for each layer
    """
    vocab_size: int
    d_model: int
    n_layers: int
    tie_weights: bool = True
    dropout: float = 0.0

    def setup(self):
        self.tok_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.rnn = LSTMStack(hidden_size=self.d_model, n_layers=self.n_layers, dropout=self.dropout)
        self.final_ln = nn.LayerNorm()
        if not self.tie_weights:
            self.out_proj = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx, *, init_state=None, deterministic: bool = True):
        # Embedding lookup
        x = self.tok_embed(idx)  # (B, T, D)

        # Run stacked LSTM
        y, final_state = self.rnn(x, init_state=init_state, deterministic=deterministic)  # y: (B, T, D)

        # Stabilize with LayerNorm before projection
        y = self.final_ln(y)

        # Project to logits
        if self.tie_weights:
            # Weight tying with the input embedding table E (V, D)
            E = self.tok_embed.embedding
            logits = jnp.einsum("btd,vd->btv", y, E)
        else:
            logits = self.out_proj(y)

        return logits, final_state
