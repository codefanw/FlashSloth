import torch
import torch.nn as nn
import math

class LearnableToken(nn.Module):
    def __init__(self, token_num, token_dim):
        super(LearnableToken, self).__init__()
        self.token_num = token_num
        self.token_dim = token_dim
        self.token = nn.Parameter(torch.Tensor(self.token_num, self.token_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.token, a=math.sqrt(5))

    def forward(self):
        return self.token
