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
    
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'.
    
    Uses sine and cosine functions of different frequencies to encode positions.
    This is deterministic and doesn't require learning.
    
    Args:
        d_model: Hidden size D.
        max_len: Maximum sequence length.
    """
    d_model: int
    max_len: int
    
    def setup(self):
        # Create positional encoding matrix
        position = jnp.arange(self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        self.pe = pe  # Store as instance variable, not trainable
    
    def __call__(self, x):
        """
        Args:
            x: Token embeddings of shape (B, T, D)
        Returns:
            x with positional encodings added, shape (B, T, D)
        """
        T = x.shape[1]
        return x + self.pe[:T]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from RoFormer paper.
    
    Instead of adding position info, RoPE rotates the query and key vectors
    based on their positions. This is applied inside attention layers.
    
    Args:
        d_head: Dimension per attention head.
        max_len: Maximum sequence length.
    """
    d_head: int
    max_len: int
    
    def setup(self):
        # Compute rotation angles
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.d_head, 2) / self.d_head))
        position = jnp.arange(self.max_len)
        # Shape: (max_len, d_head//2)
        freqs = jnp.outer(position, inv_freq)
        # Store sin and cos
        self.cos = jnp.cos(freqs)
        self.sin = jnp.sin(freqs)
    
    def rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    
    def __call__(self, q, k):
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (B, T, n_heads, d_head)
            k: Key tensor of shape (B, T, n_heads, d_head)
        Returns:
            Rotated q and k with same shapes
        """
        T = q.shape[1]
        cos = self.cos[:T, jnp.newaxis, :]  # (T, 1, d_head//2)
        sin = self.sin[:T, jnp.newaxis, :]  # (T, 1, d_head//2)
        
        # Expand for all dimensions
        cos = jnp.repeat(cos, 2, axis=-1)  # (T, 1, d_head)
        sin = jnp.repeat(sin, 2, axis=-1)  # (T, 1, d_head)
        
        # Apply rotation
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot
    
class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.
    
    Components:
    - Token embeddings: maps token ids to D-dim vectors
    - Positional embeddings: adds/encodes position information
    - N stacked decoder blocks with causal self-attention
    - Final LayerNorm
    - Output projection
    
    Args:
        vocab_size: Vocabulary size V.
        d_model: Hidden size D.
        n_layers: Number of decoder blocks.
        n_heads: Attention heads per block.
        max_len: Maximum supported sequence length.
        pos_encoding_type: Type of positional encoding to use.
            Options: 'learned' (default), 'sinusoidal', 'rotary', 'none'
    """
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    pos_encoding_type: str = 'learned'  # NEW PARAMETER
    
    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        
        # Positional encoding based on type
        if self.pos_encoding_type == 'learned':
            # Original: Learned positional embeddings
            self.positional_embed = self.param(
                "positional_embed",
                nn.initializers.normal(stddev=0.02),
                (self.max_len, self.d_model)
            )
        elif self.pos_encoding_type == 'sinusoidal':
            # Sinusoidal encoding (not learned)
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=self.d_model,
                max_len=self.max_len
            )
        elif self.pos_encoding_type == 'rotary':
            # RoPE: applied inside attention, no encoding here
            # We'll pass this info to decoder blocks
            pass
        elif self.pos_encoding_type == 'none':
            # No positional encoding
            pass
        else:
            raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}")
        
        # Stack of decoder blocks
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio
            ) for _ in range(self.n_layers)
        ]
        
        # Final LayerNorm before projecting to logits
        self.layerNorm_final = nn.LayerNorm()
        
        # Output head
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)
    
    def __call__(self, idx):
        """Forward pass (causal-only).
        
        Args:
            idx: Token ids of shape (B, T), dtype int32/int64.
        Returns:
            logits: (B, T, V) unnormalized vocabulary scores.
        """
        B, T = idx.shape
        
        # Token embeddings -> (B, T, D)
        x = self.tok_embed(idx)
        
        # Add positional information based on type
        if self.pos_encoding_type == 'learned':
            x = x + self.positional_embed[:T]
        elif self.pos_encoding_type == 'sinusoidal':
            x = self.pos_encoder(x)
        elif self.pos_encoding_type == 'rotary':
            # RoPE is handled inside attention, nothing to add here
            pass
        elif self.pos_encoding_type == 'none':
            # No positional encoding
            pass
        
        # Build attention mask: strictly causal
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal
        
        # Run the stack of decoder blocks
        for blk in self.blocks:
            x = blk(x, mask=mask)
        
        # Final LayerNorm before output projection
        x = self.layerNorm_final(x)
        
        # Output projection to logits
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