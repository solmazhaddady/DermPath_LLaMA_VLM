"""
Perceiver-based Slide-Level Aggregation Model

This module implements the slide-level aggregation architecture used for
WSI classification. Patch-level CTransPath features are enriched with
spatial positional embeddings and aggregated using a Perceiver Resampler.
The resulting slide representation is used for classification.

Components:
- Positional MLP for (x, y) coordinate encoding
- Perceiver Resampler for global feature aggregation
- Mean pooling over latent tokens
- Linear classification head (defined in training script)

Used for:
- Final diagnosis classification (BCC, SCC, No Malignancy)
- Subtype multi-label classification (shared backbone)

Author: Solmaz Haddady
"""

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, coords):
        return self.mlp(coords)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
    def forward(self, x):
        return self.net(x)

class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=96, heads=16):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q  = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out= nn.Linear(inner, dim, bias=False)
    def forward(self, media, latents):
        b, m, n, d = media.shape
        l = latents.shape[2]
        media  = self.norm_media(media)
        latents= self.norm_latents(latents)
        q = self.to_q(latents).view(b, m, l, self.heads, -1).transpose(2,3)
        k, v = self.to_kv(media).chunk(2, dim=-1)
        k = k.view(b, m, n, self.heads, -1).transpose(2,3)
        v = v.view(b, m, n, self.heads, -1).transpose(2,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(2,3).contiguous().view(b, m, l, -1)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(self, dim_feats=768, dim_model=1536, dim_head=96, num_heads=16, num_layers=6, num_latents=640):
        super().__init__()
        self.linear   = nn.Linear(dim_feats, dim_model)
        self.media_pos= nn.Parameter(torch.randn(1,1,dim_model))
        self.latents  = nn.Parameter(torch.randn(num_latents, dim_model))
        self.layers = nn.ModuleList([
            nn.ModuleList([PerceiverAttention(dim_model, dim_head, num_heads), FeedForward(dim_model)])
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)
    def forward(self, x):                   # x: [B,T,768]
        x = self.linear(x).unsqueeze(1)     # [B,1,T,1536]
        x = x + self.media_pos              # broadcast
        B = x.size(0)
        latents = self.latents.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1) # [B,1,L,1536]
        for attn, ff in self.layers:
            latents = latents + attn(x, latents)
            latents = latents + ff(latents)
        return self.norm(latents)           # [B,1,L,1536]

class PerceiverResamplerClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.pos_encoder = PositionalEncoder()
        self.resampler   = PerceiverResampler()
        self.classifier  = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, num_classes)
        )
    def forward(self, feats, mask):
        B, T, D = feats.shape
        device = feats.device
        # simple 1D positional scaffold based on sequence index (keeps original intent)
        x_pos = torch.linspace(0,1,steps=T, device=device).repeat(B,1)
        y_pos = torch.zeros_like(x_pos)
        coords = torch.stack([x_pos,y_pos], dim=-1)        # [B,T,2]
        pos_emb = self.pos_encoder(coords.view(-1,2)).view(B,T,D)
        feats = feats + pos_emb
        latents = self.resampler(feats)                    # [B,1,L,1536]
        latents = latents.mean(dim=2).squeeze(1)           # [B,1536]  MEAN PPOLING 
        logits  = self.classifier(latents)                 # [B,C]
        return logits
