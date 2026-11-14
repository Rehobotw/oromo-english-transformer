import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import warnings

from translator.config import TransformerConfig, get_weight_file_path
from translator.model import build_transformer
from translator.data import get_dataset
from translator.inference import greedy_decode


# ---------------------------------------------------------------------------
# Validation: run greedy_decode on a few examples and print results
# ---------------------------------------------------------------------------
def run_validation(model, val_dataloader, source_tokenizer, target_tokenizer,
                   max_len, device, print_fn, global_step, writer, num_examples=2):
    """Evaluate the model on a handful of validation examples and print
    the source / expected / predicted text to the console.

    This is called once per epoch during training so we can eyeball progress.
    """
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask  = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, 'Batch size must be 1 for validation'

            predicted_ids = greedy_decode(
                model, encoder_input, encoder_mask,
                source_tokenizer, target_tokenizer, max_len, device
            )

            source_text    = batch['source_text'][0]
            expected_text  = batch['target_text'][0]
            predicted_text = target_tokenizer.decode(predicted_ids.detach().cpu().numpy().tolist())

            print_fn('-' * console_width)
            print_fn(f'SOURCE:    {source_text}')
            print_fn(f'EXPECTED:  {expected_text}')
            print_fn(f'PREDICTED: {predicted_text}')

            count += 1
            if count == num_examples:
                break

    # Uncomment when you add torchmetrics:
    # writer.add_scalar('val/bleu', bleu_score, global_step)


# ---------------------------------------------------------------------------
# Build the model
# ---------------------------------------------------------------------------
def get_model(config: TransformerConfig, source_vocab_size: int, target_vocab_size: int):
    """Construct a Transformer from config values."""
    return build_transformer(
        source_vocab_size = source_vocab_size,
        target_vocab_size = target_vocab_size,
        source_seq_len    = config.seq_len,
        target_seq_len    = config.seq_len,
        model_dim         = config.hidden_dim,
        num_layers        = config.num_layers,
        num_heads         = config.num_heads,
        dropout_rate      = config.dropout_rate,
        ff_dim            = config.d_ff,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(config: TransformerConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)

    model = get_model(config,
                      source_tokenizer.get_vocab_size(),
                      target_tokenizer.get_vocab_size()).to(device)

    writer    = SummaryWriter(config.experiment_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    start_epoch = 0
    global_step = 0

    # 1. UPDATED: Resume from a checkpoint ONLY if the file actually exists
    if config.preload:
        checkpoint_path = get_weight_file_path(config, config.preload)
        if Path(checkpoint_path).exists():
            print(f'Loading checkpoint: {checkpoint_path}')
            state = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            # If the checkpoint was saved at the end of an epoch, start at the next epoch;
            # if it was saved mid-epoch, stay on the current epoch.
            if state.get('end_of_epoch', True):
                start_epoch = state['epoch'] + 1
            else:
                start_epoch = state['epoch']
        else:
            print(f'No checkpoint found at {checkpoint_path}, starting training from scratch.')

    # Ignore padding tokens in the loss so they don't skew gradients
    loss_fn = nn.CrossEntropyLoss(
        ignore_index   = source_tokenizer.token_to_id('[PAD]'),
        label_smoothing= 0.1
    ).to(device)

    save_every_n_steps = 500  # Adjust to 500 or 1000 as needed

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)   # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)   # (batch, seq_len)
            encoder_mask  = batch['encoder_mask'].to(device)    # (batch, 1, 1, seq_len)
            decoder_mask  = batch['decoder_mask'].to(device)    # (batch, 1, seq_len, seq_len)
            label         = batch['label'].to(device)           # (batch, seq_len)

            encoder_output   = model.encode(encoder_input, encoder_mask)          # (batch, seq_len, model_dim)
            decoder_output   = model.decode(encoder_output, encoder_mask,
                                            decoder_input, decoder_mask)          # (batch, seq_len, model_dim)
            projected_output = model.project(decoder_output)                      # (batch, seq_len, vocab_size)

            # Flatten batch and seq_len into one dimension for the loss function
            loss = loss_fn(
                projected_output.view(-1, target_tokenizer.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({'loss': f'{loss.item():.3f}'})

            writer.add_scalar('train/loss', loss.item(), global_step)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # 2. ADDED: Save a rolling 'latest' checkpoint every N steps inside the batch loop
            if global_step % save_every_n_steps == 0:
                latest_path = get_weight_file_path(config, 'latest')
                torch.save({
                    'epoch':                epoch,
                    'end_of_epoch':         False,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step':          global_step,
                }, latest_path)
                batch_iterator.write(f'Step {global_step}: Rolling checkpoint saved to {latest_path}')

        writer.flush()

        # Validate at the end of every epoch
        run_validation(
            model, val_dataloader, source_tokenizer, target_tokenizer,
            config.seq_len, device,
            print_fn    = lambda msg: batch_iterator.write(msg),
            global_step = global_step,
            writer      = writer,
        )

        # 3. UPDATED: Save both the numbered epoch checkpoint AND update 'latest'
        checkpoint_data = {
            'epoch':                epoch,
            'end_of_epoch':         True,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step':          global_step,
        }
        checkpoint_path = get_weight_file_path(config, f'{epoch:02d}')
        latest_path     = get_weight_file_path(config, 'latest')

        torch.save(checkpoint_data, checkpoint_path)
        torch.save(checkpoint_data, latest_path)
        print(f'Epoch checkpoint saved: {checkpoint_path} (and updated {latest_path})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = TransformerConfig()
    train_model(config)