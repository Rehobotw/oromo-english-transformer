# Changes Log — Oromo ↔ English Transformer

> This document explains every change made to the codebase in plain language.
> It is written for anyone reading the code, not just the original author.

---

## Why these changes were made

The original code had the right ideas but several functions were placed in the
wrong files, some were duplicated, and there were a handful of Python syntax
bugs that would have prevented the code from running at all. The goal of this
refactor was to give every function a single, logical home, fix the bugs, and
make variable and method names clearly describe what they do.

**No advanced abstractions were introduced.** Everything is still plain
Python classes and functions, just better organised and named.

---

## Overview of module responsibilities (the agreed architecture)

| File | What lives here |
|---|---|
| `config.py` | All settings in one place (hyperparameters, file paths, language codes) |
| `data.py` | Everything to do with loading, tokenising, and batching data |
| `model.py` | Every neural network layer, plus the `causal_mask` helper |
| `inference.py` | Generating translations with a trained model |
| `train.py` | The training loop only — imports from the modules above |
| `tests/test_model.py` | Automated tests that verify the model behaves correctly |

---

## `src/translator/config.py`

### What changed

**1. Dataclass fields were broken.**
The original file had trailing commas after string values like `lang_src='en',`.
In Python a trailing comma makes a value into a tuple, so `lang_src` was
actually `('en',)` (a tuple) instead of `'en'` (a string).  All commas were
removed and the fields were given proper type annotations (`str`, `int`, etc.)
so they are easy to understand at a glance.

**2. `__post_init__` checked the wrong attribute name.**
The dataclass field is called `lr` but the validation code checked
`self.learning_rate`, which does not exist and would crash on startup.
Fixed to check `self.lr`.

**3. Language codes are now correct and commented.**
`lang_src = 'en'` (English, the source language) and
`lang_tgt = 'orm'` (Afaan Oromo, the target language).
A comment explains that you can swap these two values to translate
in the opposite direction (Oromo → English) — the model architecture
is the same either way.

**4. `get_weight_file_path` was renamed and fixed.**
The old function was called `get_weight_file_pathJ` (a typo) and
used dictionary access (`config['model_folder']`) even though config is
now a dataclass.  It was renamed to `get_weight_file_path` and updated
to use attribute access (`config.model_folder`).

**5. Dead commented-out code was removed.**
The old `get_config()` function that returned a plain dictionary was left
commented out.  It served no purpose now that we have a proper dataclass.

---

## `src/translator/model.py`

### What changed

**1. `causal_mask` was moved here.**
It was scattered across `train.py` and `data.py`.  It belongs in `model.py`
because it is an attention mechanism — a model concern, not a data concern.
Both `data.py` and `train.py` now import it from here.

**2. Positional Encoding had a bug: cosine was missing.**
The original code applied `torch.sin` to *both* even and odd embedding
dimensions.  The correct formula (from the "Attention Is All You Need"
paper) is `sin` for even dimensions and `cos` for odd dimensions.
Line `pe[:, 1::2] = torch.cos(...)` was corrected.

**3. Typo: `unsqueeze` was spelled `unsequeeze`.**
This would crash at runtime.

**4. `Transformer` had a name collision.**
The class had an attribute called `self.decoder` (the Decoder module) **and**
a method also called `decoder()`.  Python cannot have both — the method would
shadow the attribute.  The method was renamed to `decode()`, which is also
what `train.py` was already trying to call (`model.decode(...)`).

**5. `self.d_v` does not exist.**
`MultiHeadAttention` computes `head_dim = model_dim // num_heads` for both
keys and values (they are the same size).  The original code used the
undefined name `self.d_v` when reshaping the value tensor.  Fixed to
`self.head_dim`.

**6. `nn.Module([...])` → `nn.ModuleList([...])`.**
In `DecoderBlock`, the residual connections list was created with `nn.Module`
instead of `nn.ModuleList`.  PyTorch requires `nn.ModuleList` so that it
registers the child modules for parameter tracking.

**7. `self.layer` → `self.layers` in `Decoder.forward`.**
A simple typo that would cause an `AttributeError` at training time.

**8. `key` tensor was computed from `query` instead of `key`.**
In `MultiHeadAttention.forward`, the reshape step for `key` mistakenly
called `query.view(...)` instead of `key.view(...)`.

**9. Better variable names throughout.**
Old → New:

| Old name | New name | Why |
|---|---|---|
| `w_q` | `query_proj` | Describes what the projection does |
| `w_k` | `key_proj` | Describes what the projection does |
| `w_v` | `value_proj` | Describes what the projection does |
| `w_o` | `output_proj` | Describes what the projection does |
| `d_k` | `head_dim` | Clearer: dimension per attention head |
| `h` | `num_heads` | Reads naturally |
| `pe[:,0::2]` comment | Inline comments added | Explains sin/cos split |
| `src_embed` | `source_embed` | Full words, not abbreviations |
| `tgt_embed` | `target_embed` | Full words, not abbreviations |

---

## `src/translator/data.py`

### What changed

**1. `get_all_sentences` was added.**
This function existed in `train.py` only, but `data.py` called it — so
`data.py` would crash immediately because the function was not in scope.
It now lives in `data.py` where it is used.

**2. Tokenizer imports were wrong.**
The original imports `from tokenizers import tokenizers` and
`from tokenizer.trainers import WordLevelTrainer` are both incorrect
(wrong module paths).  The correct imports are:
```python
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
```

**3. Dataset changed from `opus_books` to `allenai/wmt22_african`.**
The training dataset for this project is the WMT22 African language dataset
hosted by the Allen Institute for AI.  The Hugging Face identifier is
`allenai/wmt22_african` with the `eng-orm` subset (English ↔ Afaan Oromo).

**4. `__getitem__` was named `_getitem_` (one underscore each side).**
Python's special methods need *double* underscores: `__getitem__`.  With only
single underscores it is treated as a regular method and PyTorch's DataLoader
would silently skip it, returning empty batches.

**5. `torch.cat{...}` → `torch.cat([...]).**
`torch.cat` takes a list, not a set literal.  Using `{}` is a Python syntax
error.  Fixed all three occurrences.

**6. `self.seq_len` was never stored.**
`seq_len` was received as a constructor argument but never assigned to
`self.seq_len`, so `__getitem__` would crash with `AttributeError`.

**7. Better variable names.**
`ds` → `dataset`, `src_lang` → `source_lang`, `tgt_lang` → `target_lang`,
`enc_input_tokens` → `source_token_ids`, `enc_num_padding_tokens` →
`source_padding`, etc.

**8. `get_ds` renamed to `get_dataset`.**
The function name now reads as a complete English word instead of an abbreviation.

---

## `src/translator/inference.py`

### What changed

**1. `greedy_decode` was moved here from `train.py`.**
Generating a translation is an inference task, not a training task.
`train.py` now imports `greedy_decode` from here.

**2. The old file ran code at import time.**
The original file called `get_ds`, loaded a model, and ran `run_validation`
the moment you imported the module.  This is wrong — importing a module
should never have side effects.  The file now only defines functions.

**3. `translate()` was added.**
A clean, single-function API for the web app:
```python
result = translate("Hello world", config, checkpoint_epoch='19')
```
It handles tokenising the input, running `greedy_decode`, and decoding the
output token ids back into a string.

**4. `networkx` import was removed.**
The original imported `from networkx.convert import to_edgelist`, which has
nothing to do with translation.

**5. `from torch.nn import nn` → `import torch.nn as nn`.**
The original import was syntactically wrong.

---

## `train.py`

### What changed

**1. Duplicate functions removed.**
`get_all_sentences`, `get_or_build_tokenizer`, `get_ds`, and `greedy_decode`
were all defined in `train.py` **and** in `data.py` / `inference.py`.
The copies in `train.py` have been removed.  `train.py` now imports them:
```python
from translator.data import get_dataset
from translator.inference import greedy_decode
```

**2. Garbled line 15 removed.**
Line 15 contained a corrupted string that mixed an import statement with a
function call:
```
fron tokenizer.pre_tokenrun_validation(...)izers import Whitespace
```
This was removed — the import it was attempting is now handled by `data.py`.

**3. `config['key']` → `config.key` throughout.**
Config is now a dataclass, so you access fields with a dot, not square brackets.

**4. `optimizer.state_dic()` → `optimizer.state_dict()`.**
A typo that would crash when saving a checkpoint.

**5. `val_dataloader` variable reference fixed.**
The training loop used `val_dataloader` but the variable was declared as
`val_dataloaader` (extra `a`) at the top of the function.

**6. `run_validation` kept here.**
`run_validation` is called inside the training loop, so it belongs in
`train.py`.  It imports `greedy_decode` from `inference.py`.

**7. `get_model` kept here.**
This small helper wraps `build_transformer` with the config values and
is used only during training, so it stays in `train.py`.

**8. Better variable names.**
`train_dataloaader` → `train_dataloader`, `decoder_ouput` →
`decoder_output`, `proj_outuput` → `projected_output`, `global_state` →
`global_step` (was inconsistently named), etc.

---

## `tests/test_model.py`

### What changed

**1. `run_validation` was removed entirely.**
A copy of the `run_validation` function lived in the test file.
Tests should never contain production logic — they should only
call the production code and check that it behaves correctly.

**2. Four proper tests were added.**

| Test | What it checks |
|---|---|
| `test_encoder_output_shape` | Encoder output has the expected tensor shape |
| `test_causal_mask_blocks_future_positions` | No position can see tokens that come after it |
| `test_encoder_mask_hides_padding` | Padding positions produce a 0 in the mask |
| `test_checkpoint_save_and_load` | Weights saved to disk reload identically |

These tests can be run with `pytest tests/` or directly with
`python tests/test_model.py`.

---

## Summary of function movements

| Function | Was in | Now in |
|---|---|---|
| `get_all_sentences` | `train.py` only (missing from `data.py`) | `data.py` |
| `get_or_build_tokenizer` | `train.py` + `data.py` (duplicate) | `data.py` only |
| `get_ds` / `get_dataset` | `train.py` + `data.py` (duplicate) | `data.py` only |
| `greedy_decode` | `train.py` | `inference.py` |
| `causal_mask` | `train.py` + `data.py` (duplicate) | `model.py` |
| `translate` | did not exist | `inference.py` |
| `run_validation` | `train.py` + `tests/` (duplicate) | `train.py` only |
| `get_model` | `train.py` | `train.py` (kept, it's used only here) |
| `get_weight_file_path` | `config.py` (misspelled) | `config.py` (fixed) |
