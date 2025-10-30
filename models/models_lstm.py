import jax
import jax.numpy as jnp
import flax.linen as nn

class LSTMStack(nn.Module):
    hidden_size: int
    n_layers: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, init_state=None, deterministic: bool = True):
        B, T, _ = x.shape
        H = self.hidden_size
        cells = [nn.LSTMCell(name=f"lstm_{i}", hidden_size=H) for i in range(self.n_layers)]

        if init_state is None:
            h = jnp.zeros((self.n_layers, B, H), dtype=x.dtype)
            c = jnp.zeros((self.n_layers, B, H), dtype=x.dtype)
        else:
            h, c = init_state

        def step(carry, x_t):
            h_layers, c_layers = carry
            inp = x_t
            new_hs, new_cs = [], []
            for i, cell in enumerate(cells):
                (h_i, c_i), out = cell((h_layers[i], c_layers[i]), inp)
                out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
                new_hs.append(h_i); new_cs.append(c_i)
                inp = out
            h_stack = jnp.stack(new_hs, axis=0)
            c_stack = jnp.stack(new_cs, axis=0)
            return (h_stack, c_stack), inp

        (h_final, c_final), y_seq = jax.lax.scan(step, (h, c), jnp.swapaxes(x, 0, 1))
        y = jnp.swapaxes(y_seq, 0, 1)
        return y, (h_final, c_final)

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
            logits = self.out_proj(y)
        return logits, final_state
