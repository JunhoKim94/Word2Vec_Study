import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Negative_Sampling(nn.Module):
    def __init__(self, vocab_size, projection):
        super(Negative_Sampling, self).__init__()
        self.Embedding = nn.Embedding(vocab_size, projection)
        self.N_Embdding = nn.Embedding(vocab_size , projection)

    def forward(self, x, sampled):
        '''
        x = (N, 1) Batch x 1
        sampled = (N, sampled(k) * skip_size + 1)
        '''
        self.x = x
        
        #N x 1 x projection
        out = self.Embedding(self.x)
        out = out.unsqueeze(1)
        #N x sampled x projection
        vec = self.N_Embdding(sampled)
        #N x sampled
        output = torch.sum(out * vec, dim = 2)

        return output