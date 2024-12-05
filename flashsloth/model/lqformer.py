import torch
import torch.nn as nn
import numpy as np
import math

class lqformerattention(nn.Module):
    def __init__(self, embed_dim, num_heads, down_dim, up_dim):
        super().__init__()
        self.num_heads = num_heads
        self.down_dim = down_dim
        self.embed_dim = embed_dim
        self.down_head_dim = down_dim // num_heads
        self.head_dim = embed_dim // num_heads
        self.up_dim = up_dim
        self.q_proj = nn.Linear(self.down_dim, self.down_dim, bias=True)
        self.k_proj = nn.Linear(self.down_dim, self.down_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        

    def forward(self, query, key, value, attention_mask=None):
        bsz, q_len, _ = query.size()
        k_len = key.size(1)
        v_len = value.size(1)

        query = self.q_proj(query).view(bsz, q_len, self.num_heads, self.down_head_dim).transpose(1, 2)
        key = self.k_proj(key).view(bsz, k_len, self.num_heads, self.down_head_dim).transpose(1, 2)
        value = self.v_proj(value).view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(
            query.to(torch.float32), key.to(torch.float32).transpose(2, 3)
        ) / math.sqrt(self.down_head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e4)
            attn_weights = attn_weights + attention_mask


        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return attn_output, attn_weights
    
class LQFormerLayer(nn.Module):
    def __init__(self, d_model, mm_model, n_heads, down_dim, up_dim):
        super(LQFormerLayer, self).__init__()
        self.t2q_attn = lqformerattention(embed_dim=down_dim, num_heads=n_heads, down_dim=down_dim, up_dim=up_dim)
        self.i2q_attn = lqformerattention(embed_dim=d_model, num_heads=n_heads, down_dim=down_dim, up_dim=up_dim)
        self.ln_text = nn.LayerNorm(down_dim)
        self.ln_q = nn.LayerNorm(down_dim)
        self.ln_kv = nn.LayerNorm(down_dim)
        self.n_heads = n_heads


    def forward(self, learnable_tokens, image_tokens, image_tokens_down, text_tokens, text_mask=None):
        # Down-project learnable tokens and text tokens
        
        # Residual connection for learnable tokens before self-attention
        residual_learnable = learnable_tokens
        
        # Layer norm
        learnable_tokens = self.ln_q(learnable_tokens)
        text_tokens = self.ln_text(text_tokens)
        batch_size = learnable_tokens.size(0)   
        if text_mask is not None:
            attention_mask = text_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.repeat(1, self.n_heads, learnable_tokens.size(1), 1)
        else:
            attention_mask = None
        attn_output, _ = self.t2q_attn(query=learnable_tokens, key=text_tokens, value=text_tokens, attention_mask=attention_mask)
        
        # Cross-attention: learnable tokens query image tokens
        image_tokens_down = self.ln_kv(image_tokens_down)
        attn_output, attention_map = self.i2q_attn(query=attn_output, key=image_tokens_down, value=image_tokens, attention_mask=None)
        
        attention_map = torch.mean(attention_map, dim=1)
        return attn_output, attention_map

class LQFormer(nn.Module):
    def __init__(self, config, num_layers=1):
        super(LQFormer, self).__init__()
        self.mm_model = config.hidden_size
        self.d_model = 1152
        self.down_dim = 576
        self.down_projector_learnable_text = nn.Linear(self.mm_model, self.down_dim, bias=True)
        self.down_projector_image = nn.Linear(self.d_model, self.down_dim, bias=True)
        self.layers = nn.ModuleList([LQFormerLayer(mm_model=self.mm_model, d_model = 1152, n_heads=config.num_attention_heads, down_dim = 576, up_dim = 2560) for _ in range(num_layers)])
        self.up_projector = nn.Linear(self.d_model, self.mm_model)

    def forward(self, learnable_tokens, image_tokens, text_tokens, text_mask=None):
        learnable_tokens_down = self.down_projector_learnable_text(learnable_tokens)
        text_tokens_down = self.down_projector_learnable_text(text_tokens)
        image_tokens_down = self.down_projector_image(image_tokens)
        # Pass through the layers
        for layer in self.layers:
            residual = learnable_tokens
            learnable_tokens, attention_map = layer(learnable_tokens_down, image_tokens, image_tokens_down, text_tokens_down, text_mask)
            learnable_tokens = self.up_projector(learnable_tokens)
            learnable_tokens = residual + learnable_tokens
        return learnable_tokens
