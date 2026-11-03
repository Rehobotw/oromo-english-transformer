import torch
from pathlib import Path
from translator.config import TransformerConfig, get_weight_file_path
from translator.model import Transformer, causal_mask, build_transformer
from translator.data import get_or_build_tokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Greedy decode: generate a translation one token at a time
# ---------------------------------------------------------------------------
def greedy_decode(model, source_tokens, source_mask, source_tokenizer, target_tokenizer,
                  max_len, device):
    """Run the model in inference mode and return predicted token ids.

    'Greedy' means we always pick the single highest-probability token at each
    step.  It is fast but not always optimal (beam search would do better).
    """
    sos_id = target_tokenizer.token_to_id('[SOS]')
    eos_id = target_tokenizer.token_to_id('[EOS]')

    # Encode the source once and reuse the result for every decoding step
    encoder_output = model.encode(source_tokens, source_mask)

    # Start the decoder with just the [SOS] token
    decoder_input = torch.empty(1, 1).fill_(sos_id).type_as(source_tokens).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break   # stop if we have hit the max length

        target_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, target_mask)

        # Take the output at the last time-step and project to vocabulary
        token_probabilities = model.project(decoder_output[:, -1])
        _, next_token = torch.max(token_probabilities, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source_tokens).fill_(next_token.item()).to(device)
        ], dim=1)

        if next_token.item() == eos_id:
            break   # stop when the model produces [EOS]

    return decoder_input.squeeze(0)   # remove batch dimension


# ---------------------------------------------------------------------------
# High-level translate function (called by app.py)
# ---------------------------------------------------------------------------
def translate(text: str, config: TransformerConfig, checkpoint_epoch: str, device=None):
    """Translate a single source-language sentence and return the target string.

    Example:
        result = translate("Hello world", config, checkpoint_epoch='19')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizers
    raw_dataset      = load_dataset('allenai/wmt22_african', 'eng-orm', split='train')
    source_tokenizer = get_or_build_tokenizer(config, raw_dataset, config.lang_src)
    target_tokenizer = get_or_build_tokenizer(config, raw_dataset, config.lang_tgt)

    # Build the model and load the trained weights
    model = build_transformer(
        source_vocab_size = source_tokenizer.get_vocab_size(),
        target_vocab_size = target_tokenizer.get_vocab_size(),
        source_seq_len    = config.seq_len,
        target_seq_len    = config.seq_len,
        model_dim         = config.hidden_dim,
        num_layers        = config.num_layers,
        num_heads         = config.num_heads,
        dropout_rate      = config.dropout_rate,
        ff_dim            = config.d_ff,
    ).to(device)

    checkpoint_path = get_weight_file_path(config, checkpoint_epoch)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # Tokenise the input sentence
    source_token_ids = source_tokenizer.encode(text).ids
    sos_id = source_tokenizer.token_to_id('[SOS]')
    eos_id = source_tokenizer.token_to_id('[EOS]')
    pad_id = source_tokenizer.token_to_id('[PAD]')

    padding_needed = config.seq_len - len(source_token_ids) - 2
    if padding_needed < 0:
        raise ValueError('Input sentence is too long')

    source_tokens = torch.cat([
        torch.tensor([sos_id], dtype=torch.int64),
        torch.tensor(source_token_ids, dtype=torch.int64),
        torch.tensor([eos_id], dtype=torch.int64),
        torch.tensor([pad_id] * padding_needed, dtype=torch.int64),
    ]).unsqueeze(0).to(device)   # add batch dimension

    source_mask = (source_tokens != pad_id).unsqueeze(0).unsqueeze(0).int().to(device)

    with torch.no_grad():
        predicted_ids = greedy_decode(model, source_tokens, source_mask,
                                      source_tokenizer, target_tokenizer,
                                      config.seq_len, device)

    return target_tokenizer.decode(predicted_ids.detach().cpu().numpy())