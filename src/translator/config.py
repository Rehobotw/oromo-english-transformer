import os 
from dataclasses import dataclass 
import random
import torch 
import numpy as np

@dataclass
class TransfomerConfig:
    pass



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)