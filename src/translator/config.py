from dataclasses import dataclass, field
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
    learning_rate:float=3e-4
    batch_size:int=128
    num_epochs:int=10
    weight_decay:float=1e-4
 
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