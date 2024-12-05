import torch
import torch.nn as nn
import torch.nn.functional as F

class AveragePooling(nn.Module):
    def __init__(self, pooling_size=2, device='cpu'):
        super(AveragePooling, self).__init__()
        self.pooling_size = pooling_size
        self.device = device
        self.to(device)

    def forward(self, image_features):
        batch_size, num_features, dim = image_features.size()
        height = width = int(num_features ** 0.5)
        image_features = image_features.view(batch_size, height, width, dim)
        pooled_features = F.avg_pool2d(image_features.permute(0, 3, 1, 2), kernel_size=self.pooling_size)
        pooled_features = pooled_features.permute(0, 2, 3, 1)
        pooled_features = pooled_features.view(batch_size, -1, dim)
        return pooled_features

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, pooling_size=2, device='cpu',dtype=torch.float32):
        super(AttentionPooling, self).__init__()
        self.pooling_size = pooling_size
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, 1))
        # self.mlp.to(device,dtype)

    def forward(self, x):
        batch_size, n, dim = x.shape
        sqrt_n = int(n ** 0.5)
        pooling_size = self.pooling_size
        
        x = x.view(batch_size, sqrt_n, sqrt_n, dim)
        
        pooled_features = []
        for i in range(0, sqrt_n, pooling_size):
            for j in range(0, sqrt_n, pooling_size):
                region = x[:, i:i+pooling_size, j:j+pooling_size, :]
                region = region.reshape(batch_size, -1, dim)
                alpha = self.mlp(region)
                alpha = torch.softmax(alpha, dim=1)
                region_pooled = torch.sum(alpha * region, dim=1)
                pooled_features.append(region_pooled)
        output = torch.stack(pooled_features, dim=1)
        
        return output

def build_pooling(pooling_type, input_dim=None, pooling_size=2, device='cpu',dtype=torch.float32):
    if pooling_type == 'average':
        return AveragePooling(pooling_size=pooling_size, device=device)
    elif pooling_type == 'attention':
        if input_dim is None:
            raise ValueError("input_dim must be specified for attention pooling")
        return AttentionPooling(input_dim=input_dim, pooling_size=pooling_size, device=device, dtype=dtype)
    else:
        raise ValueError("Unknown pooling type: {}".format(pooling_type))

