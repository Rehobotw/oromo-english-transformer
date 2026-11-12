import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from datasets import load_dataset # HuggingFace library
import sentencepiece as spm
from typing import Any


# ---------------------------------------------------------------------------
# Helper: yield every sentence in one language from the dataset
# ---------------------------------------------------------------------------
def get_all_sentences(dataset, language):
    """Yield one sentence at a time from the given language column.
    Used to train the tokenizer without loading everything into memory."""
    for item in dataset:
        yield item['translation'][language]


# ---------------------------------------------------------------------------
# SentencePiece wrapper             
# ---------------------------------------------------------------------------
#Adapter: It is written so that the rest of the code functions normally without any modification.
class SPMEncoding:
    """Lightweight return value for SPMTokenizer.encode(), matching the
    .ids attribute of HuggingFace Encoding objects so the rest of the
    codebase can keep writing tokenizer.encode(text).ids unchanged."""

    __slots__ = ('ids',) # restricts the object to exactly the listed attributes

    def __init__(self, ids: list):
        self.ids = ids


class SPMTokenizer:
    """Thin wrapper around SentencePieceProcessor that exposes the same API
    the rest of the codebase expects:

        tokenizer.encode(text).ids          -> list[int]
        tokenizer.decode(ids)               -> str   (ids may be a numpy array)
        tokenizer.token_to_id('[PAD]')      -> int
        tokenizer.get_vocab_size()          -> int

    This keeps BilingualDataset, inference.py, and train.py completely
    unchanged — only the tokenizer builder below changes.
    """

    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))

    # --- encoding -----------------------------------------------------------

    def encode(self, text: str) -> SPMEncoding:
        """Encode text into sub-word ids (BPE).  Returns an SPMEncoding
        with an .ids attribute identical to the HuggingFace Encoding API."""
        return SPMEncoding(self.sp.encode(text, out_type=int))

    # --- decoding -----------------------------------------------------------

    def decode(self, ids) -> str:
        """Decode token ids back to text.
        Accepts Python lists, PyTorch tensors, or NumPy arrays."""
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        return self.sp.decode(ids)

    # --- vocabulary helpers -------------------------------------------------

    def token_to_id(self, piece: str) -> int:
        """Return the integer id for a piece / special token string."""
        return self.sp.piece_to_id(piece)

    def id_to_token(self, id_: int) -> str:
        """Return the piece string for a given integer id."""
        return self.sp.id_to_piece(id_)

    def get_vocab_size(self) -> int:
        """Return the total vocabusource_tokenizerlary size (including special tokens)."""
        return self.sp.get_piece_size()

    def __len__(self) -> int:
        return self.sp.get_piece_size()


# ---------------------------------------------------------------------------
# Tokenizer builder
# ---------------------------------------------------------------------------
def get_or_build_tokenizer(config, dataset, language) -> SPMTokenizer: #config, raw_dataset, config.lang_src
    """Load a saved SentencePiece BPE model, or train a new one and save it.

    Model files are named like:  tokenizer_en.model / tokenizer_orm.model

    Special token layout (fixed ids so the model and vocabulary always agree):
        0 → [UNK]   1 → [PAD]   2 → [SOS]   3 → [EOS]
    """
    tokenizer_path = Path(config.tokenizer_file.format(language))
    # SentencePieceTrainer writes <prefix>.model and <prefix>.vocab
    model_prefix   = str(tokenizer_path.with_suffix(''))

    if not tokenizer_path.exists():
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator = get_all_sentences(dataset, language),
            model_prefix      = model_prefix,
            model_type        = 'bpe',           # Byte-Pair Encoding
            vocab_size        = config.vocab_size,
            # If the corpus is too small to fill vocab_size, use however many
            # sub-words are available instead of raising an error.
            hard_vocab_limit  = False,
            # Pin special tokens to fixed ids (0-3) so every checkpoint is
            # consistent, regardless of corpus size or language.
            unk_id=0,  pad_id=1,  bos_id=2,  eos_id=3,
            unk_piece='[UNK]',
            pad_piece='[PAD]',
            bos_piece='[SOS]',
            eos_piece='[EOS]',
        )

    return SPMTokenizer(str(tokenizer_path))


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class BilingualDataset(Dataset):  #train_raw, source_tokenizer, target_tokenizer,config.lang_src, config.lang_tgt, config.seq_len
    """Wraps a HuggingFace dataset split and converts sentence pairs into
    the tensors the Transformer needs: token ids, masks, and labels."""

    def __init__(self, dataset, source_tokenizer, target_tokenizer,
                 source_lang, target_lang, seq_len):
        super().__init__()
        self.dataset           = dataset
        self.source_tokenizer  = source_tokenizer
        self.target_tokenizer  = target_tokenizer
        self.source_lang       = source_lang
        self.target_lang       = target_lang
        self.seq_len           = seq_len

        # Special token ids (looked up once, reused in every __getitem__ call)
        self.sos_token = torch.tensor([source_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([source_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([source_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        pair          = self.dataset[index]
        source_text   = pair['translation'][self.source_lang]
        target_text   = pair['translation'][self.target_lang]

        source_token_ids = self.source_tokenizer.encode(source_text).ids
        target_token_ids = self.target_tokenizer.encode(target_text).ids

        # How many [PAD] tokens do we need to reach seq_len?
        # Encoder input: [SOS] + tokens + [EOS] + padding
        # Decoder input: [SOS] + tokens + padding
        # Label:         tokens + [EOS] + padding
        source_padding = self.seq_len - len(source_token_ids) - 2   # -2 for SOS and EOS
        target_padding = self.seq_len - len(target_token_ids) - 1   # -1 for SOS only

        if source_padding < 0 or target_padding < 0:
            raise ValueError('Sentence is too long for the configured seq_len')

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(source_token_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * source_padding, dtype=torch.int64),
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(target_token_ids, dtype=torch.int64),
            torch.tensor([self.pad_token.item()] * target_padding, dtype=torch.int64),
        ])

        # The label is what the decoder should output — same as decoder_input
        # but shifted one step to the right (no SOS, ends with EOS)
        label = torch.cat([
            torch.tensor(target_token_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * target_padding, dtype=torch.int64),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # causal_mask is imported at use-time in train.py / inference.py to avoid circular imports
        from translator.model import causal_mask

        return {
            'encoder_input': encoder_input,                                    # (seq_len,)
            'decoder_input': decoder_input,                                    # (seq_len,)
            # encoder_mask: hide padding positions              shape (1, 1, seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # decoder_mask: hide padding AND future positions   shape (1, seq_len, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label':        label,                                             # (seq_len,)
            'source_text':  source_text,
            'target_text':  target_text,
        }


# ---------------------------------------------------------------------------
# Main data-loading function
# ---------------------------------------------------------------------------
def get_dataset(config):
    """Download / cache the WMT22 African eng-orm dataset, build tokenizers,
    and return ready-to-use DataLoaders.

    The dataset is bidirectional: config.lang_src and config.lang_tgt control
    the direction.  Swap them to translate Oromo -> English instead.
    """
    raw_dataset = load_dataset('allenai/wmt22_african',data_dir='eng-orm',split='train[:200000]',revision='refs/convert/parquet')

    source_tokenizer = get_or_build_tokenizer(config, raw_dataset, config.lang_src)
    target_tokenizer = get_or_build_tokenizer(config, raw_dataset, config.lang_tgt)

    # 90 / 10 train-validation split
    train_size     = int(0.9 * len(raw_dataset))
    val_size       = len(raw_dataset) - train_size
    train_raw, val_raw = random_split(raw_dataset, [train_size, val_size])

    train_dataset = BilingualDataset(train_raw, source_tokenizer, target_tokenizer,
                                     config.lang_src, config.lang_tgt, config.seq_len)
    val_dataset   = BilingualDataset(val_raw,   source_tokenizer, target_tokenizer,
                                     config.lang_src, config.lang_tgt, config.seq_len)

    # Print the longest sentences so we can tune seq_len if needed
    max_source_len = max(len(source_tokenizer.encode(item['translation'][config.lang_src]).ids)
                         for item in raw_dataset)
    max_target_len = max(len(target_tokenizer.encode(item['translation'][config.lang_tgt]).ids)
                         for item in raw_dataset)
    print(f'Longest source sentence (tokens): {max_source_len}')
    print(f'Longest target sentence (tokens): {max_target_len}')

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer
