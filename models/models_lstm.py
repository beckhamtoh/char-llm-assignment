# %%writefile models/models_lstm.py
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
        Returns:
          y: (B, T, H)
          final_state: tuple of per-layer LSTM states
        """
        B, T, _ = x.shape
        H = self.hidden_size
        cells = tuple(nn.LSTMCell(features=H, name=f"lstm_{i}") for i in range(self.n_layers))

        if init_state is None:
            # NOTE: size is keyword-only in this Flax version
            states = tuple(
                cell.initialize_carry(jax.random.PRNGKey(0), (B,), size=H)
                for cell in cells
            )
        else:
            states = init_state

        def step(carry, x_t):
            states = carry
            new_states = []
            inp = x_t
            for i, cell in enumerate(cells):
                state_i, out = cell(states[i], inp)
                out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
                new_states.append(state_i)
                inp = out
            return tuple(new_states), inp

        final_states, y_seq = jax.lax.scan(step, states, jnp.swapaxes(x, 0, 1))
        y = jnp.swapaxes(y_seq, 0, 1)  # (B, T, H)
        return y, final_states

class LSTMCharLM(nn.Module):
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
        x = self.tok_embed(idx)                # (B, T, D)
        y, final_state = self.rnn(x, init_state=init_state, deterministic=deterministic)
        y = self.final_ln(y)
        if self.tie_weights:
            E = self.tok_embed.embedding      # (V, D)
            logits = jnp.einsum('btd,vd->btv', y, E)
        else:
            logits = nn.Dense(self.vocab_size, use_bias=False)(y)
        return logits, final_state
