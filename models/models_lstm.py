# models_lstm.py
import jax
import jax.numpy as jnp
import flax.linen as nn


class LSTMStack(nn.Module):
    """
    Stacked LSTM using flax.linen.LSTMCell + jax.lax.scan.

    Args:
      hidden_size: int, features H per layer
      n_layers:    int, number of layers
      dropout:     float, dropout rate on each layer's output

    Input:
      x: (B, T, D)  embeddings

    Returns:
      y: (B, T, H)  top-layer hidden states
      final_state:  tuple of per-layer LSTM states (each has .c and .h of shape (B, H))
    """
    hidden_size: int
    n_layers: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, init_state=None, deterministic: bool = True):
        B, T, _ = x.shape
        H = self.hidden_size

        # Build per-layer cells
        cells = tuple(nn.LSTMCell(features=H, name=f"lstm_{i}") for i in range(self.n_layers))

        # Initialize states with explicit (B, H) zeros to avoid version-specific API differences
        if init_state is None:
            # Use the first cell to get the state type, then create zero states with (B, H)
            proto_state = cells[0].initialize_carry(jax.random.PRNGKey(0), ())
            def zero_state():
                return type(proto_state)(
                    c=jnp.zeros((B, H), dtype=x.dtype),
                    h=jnp.zeros((B, H), dtype=x.dtype),
                )
            states = tuple(zero_state() for _ in range(self.n_layers))
        else:
            states = init_state  # expect tuple of length n_layers

        def step(carry, x_t):
            layer_states = carry
            inp = x_t
            new_states = []
            for i, cell in enumerate(cells):
                state_i, out = cell(layer_states[i], inp)  # out: (B, H)
                out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
                new_states.append(state_i)
                inp = out
            return tuple(new_states), inp  # carry, y_t (top layer)

        # Time-major for scan, then back to batch-major
        final_states, y_time = jax.lax.scan(step, states, jnp.swapaxes(x, 0, 1))
        y = jnp.swapaxes(y_time, 0, 1)  # (B, T, H)
        return y, final_states


class LSTMCharLM(nn.Module):
    """
    Character LM: token embedding -> stacked LSTM -> LayerNorm -> projection.

    Args:
      vocab_size: int (V)
      d_model:    int (embedding dim and hidden size H)
      n_layers:   int
      tie_weights: bool (reuse embedding matrix for output projection)
      dropout:     float (dropout inside LSTM stack)
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
        # Embedding
        x = self.tok_embed(idx)  # (B, T, D)

        # LSTM stack
        y, final_state = self.rnn(x, init_state=init_state, deterministic=deterministic)  # (B, T, D)

        # Normalize then project
        y = self.final_ln(y)

        if self.tie_weights:
            E = self.tok_embed.embedding  # (V, D)
            logits = jnp.einsum("btd,vd->btv", y, E)
        else:
            logits = self.out_proj(y)

        return logits, final_state
