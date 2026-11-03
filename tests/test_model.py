import torch
from translator.config import TransformerConfig
from translator.model import build_transformer, causal_mask
from translator.inference import greedy_decode


# ---------------------------------------------------------------------------
# Shape smoke-test
# ---------------------------------------------------------------------------
def test_encoder_output_shape():
    """Encoder output must have the same (batch, seq_len, model_dim) shape as its input."""
    model_dim  = 64
    num_heads  = 4
    num_layers = 2
    ff_dim     = 128
    seq_len    = 10
    vocab_size = 100

    model = build_transformer(
        source_vocab_size = vocab_size,
        target_vocab_size = vocab_size,
        source_seq_len    = seq_len,
        target_seq_len    = seq_len,
        model_dim         = model_dim,
        num_layers        = num_layers,
        num_heads         = num_heads,
        dropout_rate      = 0.0,       # turn off dropout so output is deterministic
        ff_dim            = ff_dim,
    )
    model.eval()

    source_tokens = torch.randint(0, vocab_size, (1, seq_len))   # (batch=1, seq_len)
    source_mask   = torch.ones(1, 1, 1, seq_len).int()           # no padding

    with torch.no_grad():
        encoder_output = model.encode(source_tokens, source_mask)

    assert encoder_output.shape == (1, seq_len, model_dim), (
        f'Expected (1, {seq_len}, {model_dim}), got {encoder_output.shape}'
    )
    print('PASS  test_encoder_output_shape')


# ---------------------------------------------------------------------------
# Causal mask test
# ---------------------------------------------------------------------------
def test_causal_mask_blocks_future_positions():
    """Position i must NOT be able to attend to position j > i."""
    size = 5
    mask = causal_mask(size)   # shape (1, size, size), True = allowed

    assert mask.shape == (1, size, size)

    for row in range(size):
        for col in range(size):
            if col > row:
                assert not mask[0, row, col].item(), (
                    f'Position {row} should NOT see position {col} (future leak!)'
                )
            else:
                assert mask[0, row, col].item(), (
                    f'Position {row} SHOULD see position {col}'
                )
    print('PASS  test_causal_mask_blocks_future_positions')


# ---------------------------------------------------------------------------
# Padding mask test
# ---------------------------------------------------------------------------
def test_encoder_mask_hides_padding():
    """Padding positions (token id 0) must produce a 0 in the encoder mask."""
    pad_id = 0
    encoder_input = torch.tensor([[5, 3, 7, pad_id, pad_id]])   # last two are padding

    encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()

    # positions 3 and 4 are padding -> mask should be 0 there
    assert encoder_mask[0, 0, 0, 3] == 0
    assert encoder_mask[0, 0, 0, 4] == 0
    # positions 0, 1, 2 are real tokens -> mask should be 1
    assert encoder_mask[0, 0, 0, 0] == 1
    assert encoder_mask[0, 0, 0, 1] == 1
    assert encoder_mask[0, 0, 0, 2] == 1
    print('PASS  test_encoder_mask_hides_padding')


# ---------------------------------------------------------------------------
# Checkpoint round-trip test
# ---------------------------------------------------------------------------
def test_checkpoint_save_and_load(tmp_path):
    """Saving and reloading a checkpoint should produce identical model weights."""
    import os

    model_dim  = 32
    seq_len    = 6
    vocab_size = 50

    model = build_transformer(
        source_vocab_size = vocab_size,
        target_vocab_size = vocab_size,
        source_seq_len    = seq_len,
        target_seq_len    = seq_len,
        model_dim         = model_dim,
        num_layers        = 1,
        num_heads         = 2,
        dropout_rate      = 0.0,
        ff_dim            = 64,
    )

    checkpoint_file = tmp_path / 'test_checkpoint.pt'
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_file)

    # Load into a fresh model
    loaded_model = build_transformer(
        source_vocab_size = vocab_size,
        target_vocab_size = vocab_size,
        source_seq_len    = seq_len,
        target_seq_len    = seq_len,
        model_dim         = model_dim,
        num_layers        = 1,
        num_heads         = 2,
        dropout_rate      = 0.0,
        ff_dim            = 64,
    )
    state = torch.load(checkpoint_file, weights_only=True)
    loaded_model.load_state_dict(state['model_state_dict'])

    for (name, param), (_, loaded_param) in zip(model.named_parameters(),
                                                 loaded_model.named_parameters()):
        assert torch.allclose(param, loaded_param), f'Weight mismatch in {name}'

    print('PASS  test_checkpoint_save_and_load')


# ---------------------------------------------------------------------------
# Run all tests when executed directly
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import tempfile, pathlib
    test_encoder_output_shape()
    test_causal_mask_blocks_future_positions()
    test_encoder_mask_hides_padding()
    with tempfile.TemporaryDirectory() as tmp:
        test_checkpoint_save_and_load(pathlib.Path(tmp))
    print('\nAll tests passed.')