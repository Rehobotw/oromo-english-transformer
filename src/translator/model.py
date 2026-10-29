import torch.nn as nn
import torch

class InputEmbedding(nn.Module):
    def __init__(self,d_model,num_vocab):
        super().__init__()
        