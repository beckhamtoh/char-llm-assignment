import jax
import jax.numpy as jnp
import flax.linen as nn

class LSTMStack(nn.Module):
    hidden_size: int
    n_layers: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, init_state=None, deterministic: bool = True):
        """
        x: (B, T, D)
        init_state: tuple of per-layer LSTM states, or None to init zeros.
                    Each layer state is an nn.LSTMCell.LSTMState(c, h).
        Returns:
          y: (B, T, H)
          final_state: tuple of per-layer LSTM states
        """
        B, T, _ = x.shape
        H = self.hidden_size

        # Build per-layer cells (use `features`, not `hidden_size`)
        cells = tuple(nn.LSTMCell(features=H, name=f"lstm_{i}") for i in range(self.n_layers))

        # Initial state: tuple of LSTMState for each layer
        if init_state is None:
            # OK to use a fixed RNG here; it only zeros the carry
            states = tuple(cell.initialize_carry(jax.random.PRNGKey(0), (B,), H) for cell in cells)
        else:
            states = init_state  # expect a tuple(len=n_layers) of LSTMState

        def step(carry, x_t):
            states = carry
            new_states = []
            inp = x_t
            for i, cell in enumerate(cells):
                state_i, out = cell(states[i], inp)   # returns (new_state, y_t)
                out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
                new_states.append(state_i)
                inp = out                              # feed to next layer
            return tuple(new_states), inp

        # scan over time (time-major in, batch-major out)
        final_states, y_seq = jax.lax.scan(step, states, jnp.swapaxes(x, 0, 1))
        y = jnp.swapaxes(y_seq, 0, 1)  # (B, T, H)
        return y, final_states
