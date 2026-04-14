# mm_modules_stage2.py
import torch
import torch.nn as nn

class VisionCompressor(nn.Module):
    """
    Compress [B, 640, D] -> [B, K', D] with learnable queries via cross-attention pooling.
    Default: D=1536, K'=128, heads=16.
    """
    def __init__(self, dim=1536, k_prime=128, num_heads=16):
        super().__init__()
        self.k_prime = k_prime
        self.query = nn.Parameter(torch.randn(k_prime, dim))   # [K', D]
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        nn.init.normal_(self.query, mean=0.0, std=dim ** -0.5)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, N, D], e.g., output of PerceiverResampler
        returns: [B, K', D]
        """
        B, N, D = latents.shape
        Q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, K', D]
        out, _ = self.attn(Q, latents, latents, need_weights=False)
        return out  # [B, K', D]


class VisionAuxHead(nn.Module):
    """
    Optional: tiny aux head to keep the compressed tokens semantically grounded.
    Mean-pool over K' and predict a small label set ( FD or CD).
    """
    def __init__(self, dim=1536, num_classes=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, K', D]
        returns: logits [B, C]
        """
        pooled = z.mean(dim=1)          # [B, D]
        return self.fc(self.norm(pooled))
