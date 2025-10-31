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
import jax 
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

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from RoFormer paper.
    
    RoPE encodes absolute position with rotation matrix and incorporates
    explicit relative position dependency in self-attention formulation.
    
    Based on equation (14) and (15) from the paper:
    f(x_m, m) = R^d_{Θ,m} W x_m
    
    Args:
        d_model: Hidden size D (must be even).
        max_len: Maximum sequence length.
    """
    d_model: int
    max_len: int
    
    def setup(self):
        # Following paper: Θ = {θ_i = 10000^(-2(i-1)/d), i ∈ [1, 2, ..., d/2]}
        # This provides the long-term decay property
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.d_model, 2).astype(jnp.float32) / self.d_model))
        
        # Precompute cos and sin for all positions up to max_len
        # Shape: (max_len, d_model/2)
        position = jnp.arange(self.max_len, dtype=jnp.float32)
        freqs = jnp.outer(position, inv_freq)  # (max_len, d_model/2)
        
        # Store cos and sin - these are NOT trainable parameters
        self.cos_cached = jnp.cos(freqs)  # (max_len, d_model/2)
        self.sin_cached = jnp.sin(freqs)  # (max_len, d_model/2)
    
    def apply_rotary_embedding(self, x, seq_len):
        """Apply rotary embedding using equation (34) from the paper.
        
        The paper shows a computationally efficient realization:
        R^d_{Θ,m} x = [x1, x2, x3, x4, ...] ⊗ [cos mθ1, cos mθ1, cos mθ2, cos mθ2, ...]
                     + [-x2, x1, -x4, x3, ...] ⊗ [sin mθ1, sin mθ1, sin mθ2, sin mθ2, ...]
        
        Args:
            x: Input tensor of shape (B, T, D)
            seq_len: Sequence length T
        Returns:
            Rotated tensor of shape (B, T, D)
        """
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len]  # (T, d_model/2)
        sin = self.sin_cached[:seq_len]  # (T, d_model/2)
        
        # Repeat each element twice: [a, b, c] -> [a, a, b, b, c, c]
        # This matches equation (34) structure
        cos = jnp.repeat(cos, 2, axis=-1)  # (T, d_model)
        sin = jnp.repeat(sin, 2, axis=-1)  # (T, d_model)
        
        # Add batch dimension for broadcasting
        cos = cos[None, :, :]  # (1, T, d_model)
        sin = sin[None, :, :]  # (1, T, d_model)
        
        # Following equation (34): create the rotated version [-x2, x1, -x4, x3, ...]
        # Split into pairs and swap with negation
        x_reshape = x.reshape(x.shape[0], x.shape[1], -1, 2)  # (B, T, d_model/2, 2)
        x_rotated = jnp.stack([-x_reshape[..., 1], x_reshape[..., 0]], axis=-1)  # (B, T, d_model/2, 2)
        x_rotated = x_rotated.reshape(x.shape)  # (B, T, d_model)
        
        # Apply rotation: equation (34)
        return x * cos + x_rotated * sin

class RoPEAttention(nn.Module):
    """Multi-head Self-Attention with RoPE.
    
    Applies RoPE to queries and keys before computing attention,
    following equation (16) from the paper.
    
    Args:
        d_model: Hidden size D.
        n_heads: Number of attention heads.
        max_len: Maximum sequence length.
    """
    d_model: int
    n_heads: int
    max_len: int
    
    def setup(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads
        
        # Standard QKV projections
        self.wq = nn.Dense(self.d_model, use_bias=False)
        self.wk = nn.Dense(self.d_model, use_bias=False)
        self.wv = nn.Dense(self.d_model, use_bias=False)
        self.wo = nn.Dense(self.d_model, use_bias=False)
        
        # Precompute RoPE frequencies
        # Following paper: Θ = {θ_i = 10000^(-2(i-1)/d), i ∈ [1, 2, ..., d/2]}
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.d_model, 2).astype(jnp.float32) / self.d_model))
        position = jnp.arange(self.max_len, dtype=jnp.float32)
        freqs = jnp.outer(position, inv_freq)  # (max_len, d_model/2)
        
        # Store cos and sin as instance variables (not parameters)
        self.cos_cached = jnp.cos(freqs)  # (max_len, d_model/2)
        self.sin_cached = jnp.sin(freqs)  # (max_len, d_model/2)
    
    def apply_rotary_embedding(self, x, seq_len):
        """Apply rotary embedding using equation (34) from the paper.
        
        Args:
            x: Input tensor of shape (B, T, D)
            seq_len: Sequence length T
        Returns:
            Rotated tensor of shape (B, T, D)
        """
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len]  # (T, d_model/2)
        sin = self.sin_cached[:seq_len]  # (T, d_model/2)
        
        # Repeat each element twice: [a, b, c] -> [a, a, b, b, c, c]
        cos = jnp.repeat(cos, 2, axis=-1)  # (T, d_model)
        sin = jnp.repeat(sin, 2, axis=-1)  # (T, d_model)
        
        # Add batch dimension for broadcasting
        cos = cos[None, :, :]  # (1, T, d_model)
        sin = sin[None, :, :]  # (1, T, d_model)
        
        # Create rotated version: [-x2, x1, -x4, x3, ...]
        x_reshape = x.reshape(x.shape[0], x.shape[1], -1, 2)  # (B, T, d_model/2, 2)
        x_rotated = jnp.stack([-x_reshape[..., 1], x_reshape[..., 0]], axis=-1)  # (B, T, d_model/2, 2)
        x_rotated = x_rotated.reshape(x.shape)  # (B, T, d_model)
        
        # Apply rotation: equation (34)
        return x * cos + x_rotated * sin
    
    def __call__(self, x, mask=None):
        """
        Args:
            x: Input of shape (B, T, D)
            mask: Attention mask
        Returns:
            Output of shape (B, T, D)
        """
        B, T, D = x.shape
        
        # Linear projections
        q = self.wq(x)  # (B, T, D)
        k = self.wk(x)  # (B, T, D)
        v = self.wv(x)  # (B, T, D)
        
        # Apply RoPE to queries and keys
        q = self.apply_rotary_embedding(q, T)
        k = self.apply_rotary_embedding(k, T)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.d_head)
        
        # Apply causal mask
        if mask is not None:
            scores = jnp.where(mask, scores, -1e10)
        
        # Attention weights and output
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, D)
        output = self.wo(attn_output)
        
        return output
    
class DecoderBlock(nn.Module):
    """A single decoder block supporting both standard and RoPE attention.
    
    Args:
        d_model: Hidden size D.
        n_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        use_rope: Whether to use RoPE attention.
        max_len: Maximum sequence length (needed for RoPE).
    """
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    use_rope: bool = False
    max_len: int = 128
    
    @nn.compact
    def __call__(self, x, *, mask=None):
        # Pre-LayerNorm
        h = nn.LayerNorm()(x)
        
        # Attention sublayer
        if self.use_rope:
            # Use RoPE attention (equation 14-16 from paper)
            h = RoPEAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                max_len=self.max_len
            )(h, mask=mask)
        else:
            # Standard attention (equation 3 from paper)
            h = nn.SelfAttention(
                num_heads=self.n_heads,
                use_bias=False,
            )(h, mask=mask)
        
        x = x + h  # Residual
        
        # MLP sublayer
        h = nn.LayerNorm()(x)
        h = MLP(self.d_model, mlp_ratio=self.mlp_ratio)(h)
        x = x + h  # Residual
        
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
        use_rope = (self.pos_encoding_type == 'rotary')
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                use_rope=use_rope,
                max_len=self.max_len
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