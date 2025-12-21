import torch
import torch.nn as nn
import math

class GrokkingTransformerManual(nn.Module):
    def __init__(self, vocab_size=115, max_seq_len=10, n_layers=2, n_heads=4, d_model=128, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            DecoderBlock(n_heads, d_model, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_emb(x)
        h = h + self.pos_emb[:, :x.size(1), :]
        
        for layer in self.layers:
            h = layer(h)
            
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def predict(self, x):
        return self.forward_with_features(x)

    def forward_with_features(self, x):
        features = []
        h = self.tok_emb(x)
        h = h + self.pos_emb[:, :x.size(1), :]
        
        for layer in self.layers:
            h, block_features = layer.forward_with_features(h)
            features.extend(block_features)
            
        h = self.ln_f(h)
        logits = self.head(h)
        return logits, features

class DecoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(n_heads, d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
    def forward_with_features(self, x):
        features = []
        attn_out, attn_features = self.attn.predict(self.ln_1(x))
        x = x + attn_out
        features.extend(attn_features)
        
        # To get MLP features, we can't use nn.Sequential directly
        h = self.ln_2(x)
        h1 = self.mlp[0](h)
        features.append(h1)
        h2 = self.mlp[1](h1)
        h3 = self.mlp[2](h2)
        features.append(h3)
        h4 = self.mlp[3](h3)
        x = x + h4

        return x, features

class CausalSelfAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = torch.nn.functional.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.W_o(y)

    def predict(self, x):
        return self.forward_with_features(x)

    def forward_with_features(self, x):
        features = []
        B, T, C = x.size()
        
        q = self.W_q(x)
        features.append(q)
        k = self.W_k(x)
        features.append(k)
        v = self.W_v(x)
        features.append(v)
        
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = torch.nn.functional.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.W_o(y), features

class GrokkingTransformerStandard(nn.Module):
    def __init__(self, vocab_size=115, max_seq_len=10, n_layers=2, n_heads=4, d_model=128, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # The standard nn.TransformerDecoder expects target and memory. 
        # For decoder-only, we pass the same tensor for both.
        h = self.tok_emb(x)
        h = h + self.pos_emb[:, :x.size(1), :]
        
        # Create a causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        
        h = self.transformer_decoder(h, h, tgt_mask=tgt_mask)
        
        logits = self.head(h)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        return logits, []