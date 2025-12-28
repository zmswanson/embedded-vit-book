# lightning_modules/heads.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(x)

class CosFaceHead(nn.Module):
    """
    CosFace / AM-Softmax: cos(theta) - m for target class, scaled by s.
    """
    def __init__(self, in_dim: int, num_classes: int, s: float = 64.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = F.linear(x, W)  # [B, C], cosine similarity
        # subtract margin on target logits
        one_hot = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1.0)
        logits = logits - one_hot * self.m
        return logits * self.s

class AdaFaceHead(nn.Module):
    """
    AdaFace (adaptive margin based on feature norm).
    This is a commonly-used practical implementation pattern:
      - normalize features/weights for cosine
      - compute norm-based "margin scaler"
      - apply additive angular + additive cosine style adjustments (per paper design)
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.eps = eps

        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.W)

        # running stats of feature norms
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))
        self.register_buffer("_stats_initialized", torch.tensor(False))  # bool-ish

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # feature norm stats
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp_min(self.eps)  # [B,1]
        with torch.no_grad():
            mean = x_norm.mean()
            std = x_norm.std().clamp_min(self.eps)
            # self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * mean
            # self.batch_std  = (1 - self.t_alpha) * self.batch_std  + self.t_alpha * std

            # Bootstrap on first batch to avoid bias
            if not bool(self._stats_initialized.item()):
                self.batch_mean.copy_(mean)
                self.batch_std.copy_(std)
                self._stats_initialized.fill_(True)
            else:
                self.batch_mean.mul_(1 - self.t_alpha).add_(self.t_alpha * mean)
                self.batch_std.mul_(1 - self.t_alpha).add_(self.t_alpha * std)

        # margin scaler in [-h, h]
        margin_scaler = ((x_norm - self.batch_mean) / self.batch_std).clamp(-1, 1) * self.h  # [B,1]

        # cosine logits
        x_n = F.normalize(x, dim=1)
        W_n = F.normalize(self.W, dim=1)
        cos_t = F.linear(x_n, W_n).clamp(-1 + self.eps, 1 - self.eps)  # [B,C]

        # target-only adjustments
        theta = torch.acos(cos_t)
        one_hot = torch.zeros_like(cos_t).scatter_(1, y.view(-1, 1), 1.0)

        # adaptive angular margin (theta + m*(1+scaler))
        m_arc = self.m * (1.0 + margin_scaler)  # [B,1]
        theta_m = theta + one_hot * m_arc

        # back to cosine
        cos_t_m = torch.cos(theta_m)

        # optional: adaptive additive cosine margin too (common in AdaFace codebases)
        m_cos = self.m * (1.0 - margin_scaler)  # [B,1]
        cos_t_m = cos_t_m - one_hot * m_cos

        return cos_t_m * self.s


class ArcFaceHead(nn.Module):
    """
    ArcFace / Additive Angular Margin Softmax:
      cos(theta + m) for target class, cos(theta) otherwise, scaled by s.

    Typical defaults: s=64.0, m=0.50
    """
    def __init__(self, in_dim: int, num_classes: int, s: float = 64.0, m: float = 0.50, eps: float = 1e-6):
        super().__init__()
        self.s = s
        self.m = m
        self.eps = eps

        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.W)

        # Precompute constants used in the "phi" expression.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Optional "easy margin" handling; see forward() for version used here.
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m  # for the "non-easy" margin correction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights to get cosine similarity.
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        cos_t = F.linear(x, W).clamp(-1.0 + self.eps, 1.0 - self.eps)  # [B, C]

        # sin(theta) from cos(theta)
        sin_t = torch.sqrt((1.0 - cos_t * cos_t).clamp_min(0.0))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cos_t * self.cos_m - sin_t * self.sin_m

        # "non-easy margin" fix to keep monotonicity:
        # if cos_t <= cos(pi - m), use cos_t - sin(pi - m)*m
        phi = torch.where(cos_t > self.th, phi, cos_t - self.mm)

        # Apply phi to target logits only
        one_hot = torch.zeros_like(cos_t).scatter_(1, y.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cos_t

        return logits * self.s
