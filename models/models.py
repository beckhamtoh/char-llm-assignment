"""
Minimal decoder-only Transformer blocks in Flax/JAX, commented for learning.

The model mirrors a GPT-style architecture:
- Token embeddings + learned positional embeddings
- Stack of Pre-LayerNorm decoder blocks with causal self-attention
- Final LayerNorm
- Weight tying between input embeddings and output logits projection

Tensor shape conventions used below:
- B: batch size
- T: sequence length (time/positions)
- D: hidden size / embedding dimension (d_model)
- V: vocabulary size
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class MLP(nn.Module):
        """Transformer feed-forward network (a.k.a. MLP block).

        Structure: Dense(D -> 4D), GELU, Dense(4D -> D) by default.
        The expansion factor can be adjusted with `mlp_ratio`.

        Args:
            d_model: Hidden size D.
            mlp_ratio: Expansion factor for the intermediate hidden size.

        Input shape:  (B, T, D)
        Output shape: (B, T, D)
        """

        d_model: int
        mlp_ratio: int = 4

        @nn.compact
        def __call__(self, x):
                # Expand channel dimension (D -> hidden), apply non-linearity, project back to D.
                hidden = int(self.d_model * self.mlp_ratio)
                x = nn.Dense(hidden)(x)
                x = nn.gelu(x)
                x = nn.Dense(self.d_model)(x)
                return x

class DecoderBlock(nn.Module):
    """A single decoder block (Pre-LayerNorm + Self-Attn + MLP + residuals).

    Pre-LayerNorm improves training stability. Residual connections are used after
    attention and MLP sublayers. The attention is causal when a causal mask is passed
    (so each position can only attend to previous or current positions).

    Args:
      d_model: Hidden size D.
      n_heads: Number of attention heads.

    Input/Output shape: (B, T, D)
    """

    d_model: int
    n_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, *, mask=None):
        # Attention sublayer: Pre-LayerNorm -> Self-Attention -> Residual add
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
        )(h, mask=mask)
        x = x + h  # residual connection

        # MLP sublayer: Pre-LayerNorm -> MLP -> Residual add
        h = nn.LayerNorm()(x)
        h = MLP(self.d_model, mlp_ratio=self.mlp_ratio)(h)
        x = x + h  # residual connection
        return x

class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.

    Components:
      - Token embeddings: maps token ids to D-dim vectors
      - Learned positional embeddings: adds position information (0..T-1)
      - N stacked decoder blocks with causal self-attention
      - Final LayerNorm
      - Output projection:
          * If tie_weights=True (default), reuse token embedding matrix E to
            compute logits via x @ E^T (implemented via einsum).
          * Else, use a separate linear head to project to V logits.

    Args:
      vocab_size: Vocabulary size V.
      d_model: Hidden size D.
      n_layers: Number of decoder blocks.
      n_heads: Attention heads per block.
      max_len: Maximum supported sequence length for positional embeddings.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4

    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        # Learned positional embeddings P with shape (max_len, D)
        # We'll slice P[:T] each forward pass and add to token embeddings.
        self.positional_embed = self.param(
            "positional_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )

        # Stack of decoder blocks
        self.blocks = [DecoderBlock(d_model=self.d_model, n_heads=self.n_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.n_layers)]

        # Final LayerNorm before projecting to logits
        self.layerNorm_final = nn.LayerNorm()

        # Optional separate output head if not weight-tying
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx):
        """Forward pass (causal-only).

        Args:
          idx: Token ids of shape (B, T), dtype int32/int64.

        Returns:
          logits: (B, T, V) unnormalized vocabulary scores for next-token prediction.
        """
        B, T = idx.shape

        # Token + positional embeddings -> (B, T, D)
        x = self.tok_embed(idx) + self.positional_embed[:T]

        # Build attention mask: strictly causal (lower-triangular), no padding mask.
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal

        # Run the stack of decoder blocks
        for blk in self.blocks:
            x = blk(x, mask=mask)

        # Final LayerNorm before output projection
        x = self.layerNorm_final(x)

        # Output projection to logits over V tokens.
        logits = self.project_to_vocab(x)
        
        return logits
    

class CharLSTM(nn.Module):
    """LSTM-based character-level language model.
    
    Components:
    - Token embeddings: maps token ids to D-dim vectors
    - Stacked LSTM layers
    - Output projection to vocabulary
    
    Args:
        vocab_size: Vocabulary size V.
        d_model: Hidden size D (LSTM hidden dimension).
        n_layers: Number of LSTM layers.
    """
    vocab_size: int
    d_model: int
    n_layers: int
    
    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        
        # Stack of LSTM cells - each needs to know the hidden dimension
        self.lstm_cells = [nn.LSTMCell(features=self.d_model) for _ in range(self.n_layers)]
        
        # Output projection to vocabulary
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)
    
    def __call__(self, idx):
        """Forward pass.
        
        Args:
            idx: Token ids of shape (B, T), dtype int32/int64.
            
        Returns:
            logits: (B, T, V) unnormalized vocabulary scores for next-token prediction.
        """
        B, T = idx.shape
        
        # Token embeddings -> (B, T, D)
        x = self.tok_embed(idx)
        
        # Initialize LSTM states for all layers
        carry = []
        for layer_idx in range(self.n_layers):
            c = jnp.zeros((B, self.d_model))
            h = jnp.zeros((B, self.d_model))
            carry.append((c, h))
        
        # Process sequence step by step
        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, D)
            
            # Pass through each LSTM layer
            for layer_idx in range(self.n_layers):
                (c, h) = carry[layer_idx]
                (c, h), xt = self.lstm_cells[layer_idx](carry[layer_idx], xt)
                carry[layer_idx] = (c, h)
            
            outputs.append(xt)
        
        # Stack outputs -> (B, T, D)
        x = jnp.stack(outputs, axis=1)
        
        # Project to vocabulary -> (B, T, V)
        logits = self.project_to_vocab(x)
        
        return logits