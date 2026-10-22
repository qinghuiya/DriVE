from __future__ import annotations

import math

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f'image_size ({image_size}) must be divisible by patch_size ({patch_size}).'
            )
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'ViT expects NCHW input, got shape {tuple(x.shape)}.')
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f'ViT configured for {self.image_size}x{self.image_size} inputs, '
                f'got {tuple(x.shape[-2:])}.'
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViTTiny(nn.Module):
    """Small self-contained Vision Transformer for CIFAR-sized images.

    Defaults follow the supplementary experiment: patch size 4,
    embedding dimension 192, six encoder blocks and three attention heads.
    The implementation uses only PyTorch and therefore does not require timm.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')
        if depth <= 0:
            raise ValueError('depth must be positive.')

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(round(embed_dim * mlp_ratio)),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        # PyTorch's TransformerEncoderLayer exposes one dropout value for the
        # whole block.  Keep attention_dropout in the public signature so the
        # protocol is explicit, but reject unsupported mismatches rather than
        # silently pretending to use a different value.
        if not math.isclose(float(attention_dropout), float(dropout), abs_tol=1e-12):
            raise ValueError(
                'This dependency-free ViT uses the same dropout for attention '
                'and MLP paths; set attention_dropout equal to dropout.'
            )
        encoder_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=depth,
                norm=encoder_norm,
                enable_nested_tensor=False,
            )
        except TypeError:  # Compatibility with older supported PyTorch releases.
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=depth,
                norm=encoder_norm,
            )
        self.head_drop = nn.Dropout(classifier_dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head:
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        return self.head(self.head_drop(x[:, 0]))
