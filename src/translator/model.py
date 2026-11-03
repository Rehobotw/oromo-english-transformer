import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Upper-triangular mask
# ---------------------------------------------------------------------------
def causal_mask(size):
    """Return a square upper-triangular mask that hides future positions.
    Shape: (1, size, size)  --  0 means 'blocked', 1 means 'allowed'.
    """
    mask = torch.triu(
        torch.ones(1, size, size), 
        diagonal=1
        ).type(torch.int)
    return mask == 0   # flip: True where attention is allowed


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class InputEmbedding(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, token_ids):
        # Scale by sqrt(d_model) as in the original paper
        return self.embedding(token_ids) * math.sqrt(self.model_dim)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, seq_len: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # Build the (seq_len, model_dim) position matrix once, then freeze it
        pe = torch.zeros(seq_len, model_dim)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)          # (seq_len, 1)
        scale = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))

        pe[:, 0::2] = torch.sin(positions * scale)  # even dimensions
        pe[:, 1::2] = torch.cos(positions * scale)  # odd dimensions  

        pe = pe.unsqueeze(0)   # add batch dimension: (1, seq_len, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, model_dim)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Layer Normalisation
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1))   # learnable multiplier
        self.shift = nn.Parameter(torch.zeros(1))  # learnable bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift


# ---------------------------------------------------------------------------
# Feed-Forward Block
# ---------------------------------------------------------------------------
class FeedForwardBlock(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, dropout_rate: float):
        super().__init__()
        self.expand  = nn.Linear(model_dim, ff_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.shrink  = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        # (batch, seq_len, model_dim) -> (batch, seq_len, ff_dim) -> (batch, seq_len, model_dim)
        return self.shrink(self.dropout(torch.relu(self.expand(x))))


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert model_dim % num_heads == 0, 'model_dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim  = model_dim // num_heads   # dimension per head

        self.query_proj  = nn.Linear(model_dim, model_dim)
        self.key_proj    = nn.Linear(model_dim, model_dim)
        self.value_proj  = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)
        self.dropout     = nn.Dropout(dropout_rate)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask, dropout):
        head_dim = query.shape[-1]
        # (batch, heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_weights = dropout(attention_weights)

        return attention_weights @ value, attention_weights

    def forward(self, query_input, key_input, value_input, mask):
        query = self.query_proj(query_input)
        key   = self.key_proj(key_input)
        value = self.value_proj(value_input)

        batch_size = query.shape[0]

        # Split into heads: (batch, seq_len, model_dim) -> (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attended, self.attention_weights = MultiHeadAttention.scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )

        # Merge heads back: (batch, heads, seq_len, head_dim) -> (batch, seq_len, model_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.output_proj(attended)


# ---------------------------------------------------------------------------
# Residual Connection (Add & Norm)
# ---------------------------------------------------------------------------
class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm    = LayerNorm()

    def forward(self, x, sublayer):
        # Pre-norm variant (norm before sublayer, as commonly implemented)
        return x + self.dropout(sublayer(self.norm(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout_rate: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward   = feed_forward
        self.residuals      = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(2)])

    def forward(self, x, source_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, source_mask))
        x = self.residuals[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNorm()

    def forward(self, x, source_mask):
        for layer in self.layers:
            x = layer(x, source_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock, dropout_rate: float):
        super().__init__()
        self.self_attention  = self_attention
        self.cross_attention = cross_attention
        self.feed_forward    = feed_forward
        self.residuals       = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.residuals[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, source_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNorm()

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Projection (Decoder output -> vocabulary probabilities)
# ---------------------------------------------------------------------------
class ProjectionLayer(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.project = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.project(x), dim=-1)


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 source_embed: InputEmbedding, target_embed: InputEmbedding,
                 source_pos: PositionalEncoding, target_pos: PositionalEncoding,
                 projection: ProjectionLayer):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection = projection

    def encode(self, source_tokens, source_mask):
        x = self.source_embed(source_tokens)
        x = self.source_pos(x)
        return self.encoder(x, source_mask)

    def decode(self, encoder_output, source_mask, target_tokens, target_mask):
        x = self.target_embed(target_tokens)
        x = self.target_pos(x)
        return self.decoder(x, encoder_output, source_mask, target_mask)

    def project(self, decoder_output):
        return self.projection(decoder_output)


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------
def build_transformer(source_vocab_size: int, target_vocab_size: int,
                      source_seq_len: int, target_seq_len: int,
                      model_dim: int = 512, num_layers: int = 6,
                      num_heads: int = 8, dropout_rate: float = 0.1,
                      ff_dim: int = 2048) -> Transformer:
    """Assemble and return a fully initialised Transformer model."""

    source_embed = InputEmbedding(model_dim, source_vocab_size)
    target_embed = InputEmbedding(model_dim, target_vocab_size)

    source_pos = PositionalEncoding(model_dim, source_seq_len, dropout_rate)
    target_pos = PositionalEncoding(model_dim, target_seq_len, dropout_rate)

    encoder_blocks = []
    for _ in range(num_layers):
        encoder_blocks.append(EncoderBlock(
            MultiHeadAttention(model_dim, num_heads, dropout_rate),
            FeedForwardBlock(model_dim, ff_dim, dropout_rate),
            dropout_rate
        ))

    decoder_blocks = []
    for _ in range(num_layers):
        decoder_blocks.append(DecoderBlock(
            MultiHeadAttention(model_dim, num_heads, dropout_rate),   # self-attention
            MultiHeadAttention(model_dim, num_heads, dropout_rate),   # cross-attention
            FeedForwardBlock(model_dim, ff_dim, dropout_rate),
            dropout_rate
        ))

    encoder   = Encoder(nn.ModuleList(encoder_blocks))
    decoder   = Decoder(nn.ModuleList(decoder_blocks))
    projection = ProjectionLayer(model_dim, target_vocab_size)

    transformer = Transformer(encoder, decoder, source_embed, target_embed,
                              source_pos, target_pos, projection)

    # Xavier uniform init — keeps gradients stable at the start of training
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer