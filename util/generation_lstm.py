import jax
import jax.numpy as jnp

def generate_tokens_lstm(model, params, rng, context, length, temperature=1.0, sample=True):
    """
    model: LSTMCharLM
    context: (B, S) int32, S >= 1
    returns: (B, length) int32
    """
    context = context.astype(jnp.int32)   # NEW
    B, S = context.shape
    assert S >= 1

    logits, state = model.apply({"params": params}, context, deterministic=True)
    last_token = context[:, -1]  # (B,)

    def step(carry, _):
        rng, last_tok, state = carry
        logits_t, state = model.apply({"params": params},
                                      last_tok[:, None],
                                      init_state=state,
                                      deterministic=True)
        logits_t = logits_t[:, -1, :]  # (B, V)
        rng, sub = jax.random.split(rng)
        if sample:
            temp = temperature if (temperature and temperature > 0) else 1.0  # guard
            nxt = jax.random.categorical(sub, logits_t / temp, axis=-1)
        else:
            nxt = jnp.argmax(logits_t, axis=-1)
        nxt = nxt.astype(jnp.int32)
        return (rng, nxt, state), nxt

    (rng_final, last_token_final, state_final), toks = jax.lax.scan(
        step, (rng, last_token, state), None, length=length
    )
    return toks.transpose(1, 0)  # (B, length)
