"""TSMM ResNet50 + CBAM + TCN + MHA + BBox (paper full config)."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        avg_c = x.mean(dim=1, keepdim=True)
        max_c = x.amax(dim=1, keepdim=True)
        sa = torch.sigmoid(self.conv(torch.cat([avg_c, max_c], dim=1)))
        return x * sa


class ResNet50Backbone(nn.Module):
    """ResNet50 with CBAM inserted after layer4 (before avgpool)."""

    def __init__(self, use_cbam: bool = True, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam = CBAM(2048) if use_cbam else None
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_features = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.cbam is not None:
            x = self.cbam(x)
        x = self.avgpool(x)
        return x.flatten(1)


class VideoTSMModel(nn.Module):
    def __init__(
        self,
        use_cbam: bool = True,
        use_tcn: bool = True,
        use_trans: bool = True,
        use_bbox: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = ResNet50Backbone(use_cbam=use_cbam, pretrained=pretrained)
        in_f = self.backbone.out_features

        self.use_tcn = use_tcn
        self.use_trans = use_trans
        self.use_bbox = use_bbox

        if use_tcn:
            layers = []
            for dilation in (1, 2, 4):
                layers += [
                    nn.Conv1d(in_f, in_f, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                ]
            self.tcn = nn.Sequential(*layers)

        if use_trans:
            self.attn = nn.MultiheadAttention(in_f, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(in_f)
            self.ffn = nn.Sequential(
                nn.Linear(in_f, in_f * 4),
                nn.ReLU(),
                nn.Linear(in_f * 4, in_f),
            )
            self.norm2 = nn.LayerNorm(in_f)

        if use_bbox:
            self.bbox_mlp = nn.Sequential(
                nn.Linear(4, in_f // 4),
                nn.ReLU(),
                nn.Linear(in_f // 4, in_f // 4),
            )
            self.cls = nn.Linear(in_f + in_f // 4, 2)
        else:
            self.cls = nn.Linear(in_f, 2)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, bbox_feats: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c, h, w = x.shape
        # Frame-wise backbone forward to reduce peak memory (B*T -> B).
        frame_feats = [self.backbone(x[:, i]) for i in range(t)]
        f = torch.stack(frame_feats, dim=1).permute(0, 2, 1)

        if self.use_tcn:
            f = self.tcn(f)

        if self.use_trans:
            ft = f.permute(0, 2, 1)
            attn_out, _ = self.attn(ft, ft, ft)
            ft = self.norm1(ft + attn_out)
            ffn_out = self.ffn(ft)
            ft = self.norm2(ft + ffn_out)
            f = ft.permute(0, 2, 1)

        v = f.permute(0, 2, 1).mean(dim=1)

        if self.use_bbox:
            if bbox_feats is None:
                bbox_feats = torch.zeros(b, 4, device=v.device, dtype=v.dtype)
            bf = self.bbox_mlp(bbox_feats)
            v = torch.cat([v, bf], dim=1)

        v = self.dropout(v)
        return self.cls(v)


def build_model(config: dict | None = None, pretrained: bool = True) -> VideoTSMModel:
    cfg = config or {}
    return VideoTSMModel(
        use_cbam=cfg.get("cbam", True),
        use_tcn=cfg.get("tcn", True),
        use_trans=cfg.get("trans", True),
        use_bbox=cfg.get("bbox", True),
        pretrained=pretrained,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, create_fn=None, decay: float = 0.99):
        if create_fn is None:
            create_fn = lambda: build_model(pretrained=False)
        self.ema = create_fn().to(next(model.parameters()).device).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.ema.load_state_dict(model.state_dict(), strict=True)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            if not torch.isfinite(model_p.data).all():
                continue
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)
