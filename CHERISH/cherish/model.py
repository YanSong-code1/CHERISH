"""Hierarchical multitask MIL model used by CHERISH."""

from __future__ import annotations

import torch
from torch import nn


class GatedAttentionPool(nn.Module):
    """Gated attention pooling over tile-level feature embeddings."""

    def __init__(self, embed_dim: int, attn_dim: int, dropout: float) -> None:
        super().__init__()
        self.pre = nn.Dropout(dropout)
        self.attn_a = nn.Linear(embed_dim, attn_dim)
        self.attn_b = nn.Linear(embed_dim, attn_dim)
        self.attn_c = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        a = torch.tanh(self.attn_a(h))
        b = torch.sigmoid(self.attn_b(h))
        logits = self.attn_c(a * b).squeeze(1)
        attn = torch.softmax(logits, dim=0)
        bag = torch.sum(attn.unsqueeze(1) * x, dim=0)
        return bag, attn


class MultitaskHER2MILHier(nn.Module):
    """CHERISH hierarchical multitask MIL network.

    The polarity anchor estimates a negative-side versus
    amplification/strong-overexpression morphology axis. The positive-boundary
    module separates IHC2/FISH-amplified from IHC3 morphology along that axis.
    The primary boundary-state head estimates the three-class endpoint:
    IHC2/FISH-negative, IHC2/FISH-positive and IHC3.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.25,
        prototype_dim: int | None = None,
        num_task_b_prototypes_per_class: int = 8,
        num_task_c_prototypes_per_class: int = 8,
        use_coords: bool = True,
        use_task_c_residual: bool = True,
        family: str = "hierarchical_taskc",
        **_: object,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.is_single_task_taskc = self.family in {"single_task_taskc", "taskc_only"}
        self.embed_dim = int(embed_dim)
        self.prototype_dim = int(prototype_dim or embed_dim)
        self.use_coords = bool(use_coords)
        self.use_task_c_residual = bool(use_task_c_residual)
        self.num_task_b_prototypes_per_class = int(num_task_b_prototypes_per_class)
        self.num_task_c_prototypes_per_class = int(num_task_c_prototypes_per_class)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.coord_proj = nn.Linear(2, self.embed_dim) if self.use_coords else None

        self.shared_pool = GatedAttentionPool(self.embed_dim, attn_dim, dropout)
        self.task_a_pool = GatedAttentionPool(self.embed_dim, attn_dim, dropout)
        self.task_b_pool = GatedAttentionPool(self.embed_dim, attn_dim, dropout)
        self.task_c_pool = GatedAttentionPool(self.embed_dim, attn_dim, dropout)

        self.task_a_classifier = nn.Linear(self.embed_dim, 1)
        self.task_b_classifier = nn.Linear(self.embed_dim, 1)
        self.task_c_residual = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 3),
        )

        if self.prototype_dim == self.embed_dim:
            self.prototype_proj = nn.Identity()
        else:
            self.prototype_proj = nn.Linear(self.embed_dim, self.prototype_dim)
        self.task_b_prototypes = nn.Parameter(
            torch.randn(max(2, 2 * self.num_task_b_prototypes_per_class), self.prototype_dim) * 0.02
        )
        self.task_c_prototypes = nn.Parameter(
            torch.randn(max(3, 3 * self.num_task_c_prototypes_per_class), self.prototype_dim) * 0.02
        )

    def _normalize_coords(self, coords: torch.Tensor | None) -> torch.Tensor | None:
        if coords is None or coords.numel() == 0:
            return None
        coords = coords.to(dtype=torch.float32)
        min_xy = coords.min(dim=0).values
        max_xy = coords.max(dim=0).values
        span = (max_xy - min_xy).clamp_min(1.0)
        return (coords - min_xy) / span

    def get_task_b_prototypes(self, class_index: int) -> torch.Tensor:
        start = int(class_index) * self.num_task_b_prototypes_per_class
        end = start + self.num_task_b_prototypes_per_class
        return self.task_b_prototypes[start:end]

    def get_task_c_prototypes(self, class_index: int) -> torch.Tensor:
        start = int(class_index) * self.num_task_c_prototypes_per_class
        end = start + self.num_task_c_prototypes_per_class
        return self.task_c_prototypes[start:end]

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        if self.coord_proj is not None and coords is not None:
            h = h + self.coord_proj(self._normalize_coords(coords))

        shared_bag, shared_attn = self.shared_pool(h)
        task_a_bag, task_a_attn = self.task_a_pool(h)
        task_b_bag, task_b_attn = self.task_b_pool(h)
        task_c_bag, task_c_attn = self.task_c_pool(h)

        task_a_logit = self.task_a_classifier(task_a_bag.unsqueeze(0)).squeeze(0)
        task_b_logit = self.task_b_classifier(task_b_bag.unsqueeze(0)).squeeze(0)

        residual_logit = self.task_c_residual(torch.cat([task_c_bag, shared_bag], dim=0))
        if not self.use_task_c_residual:
            residual_logit = torch.zeros_like(residual_logit)

        if self.is_single_task_taskc:
            task_c_logit = residual_logit
            base_probs = torch.softmax(task_c_logit, dim=0)
        else:
            alpha = torch.sigmoid(task_a_logit).squeeze(0)
            beta = torch.sigmoid(task_b_logit).squeeze(0)
            base_probs = torch.stack([1.0 - alpha, alpha * beta, alpha * (1.0 - beta)])
            task_c_logit = torch.log(base_probs.clamp_min(1e-6)) + residual_logit

        out = {
            "task_a_logit": task_a_logit,
            "task_b_logit": task_b_logit,
            "task_c_logit": task_c_logit,
            "task_c_alpha_logit": task_a_logit,
            "task_c_beta_logit": task_b_logit,
            "task_c_residual_logit": residual_logit,
            "task_c_base_prob": base_probs,
        }
        if return_features:
            out.update(
                {
                    "shared_bag_features": shared_bag,
                    "task_a_bag_features": task_a_bag,
                    "task_b_bag_features": task_b_bag,
                    "task_c_bag_features": task_c_bag,
                    "instance_features": h,
                    "prototype_instance_features": self.prototype_proj(h),
                    "shared_attn": shared_attn,
                    "task_a_attn": task_a_attn,
                    "task_b_attn": task_b_attn,
                    "task_c_attn": task_c_attn,
                }
            )
        return out
