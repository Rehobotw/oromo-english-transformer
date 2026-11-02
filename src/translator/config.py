from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import random
import torch 
import os 


@dataclass
class TransfomerConfig:
    # model hpyerparameters
    hidden_dim:int=4
    num_layers:int=3
    dropout_rate:float=0.2
    
    # training hyperparameters
    lr:float=3e-4 # learning rate
    batch_size:int=8
    num_epochs:int=20
    weight_decay:float=1e-4

    seq_len=350
    d_model=512
    lang_src='en',
    lang_tgt='orm',
    model_folder='weights',
    model_filename='tmodel_',
    preload=None,
    tokenizer_file='tokeinizer_{0}.json',
    experiment_name='runs/tmodel'
 
    def __post_init__(self) -> None:
        #Validate fields and compute derived values.

        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {self.dropout_rate}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
      

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_config():
    return{
        #'batch_size':8,
        #'num_epochs':20,
        #'lr':10**-4,
        'seq_len':350,
        'd_model':512,
        'lang_src':'en',
        'lang_tgt':'it',
        'model_folder':'weights',
        'model_filename':'tmodel_',
        'preload':None,
        'tokenizer_file':'tokeinizer_{0}.json',
        'experiment_name':'runs/tmodel'
    }

def get_weight_file_pathJ(config,epoch:str):
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path('.'/model_folder/model_filename))