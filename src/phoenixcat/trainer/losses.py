# Copyright 2024 Hongyao Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..auto import auto_register_save_load


def max_margin_loss(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()


def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets.detach().cpu()] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) * (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()


_LOSS_MAPPING = {
    'ce': F.cross_entropy,
    'poincare': poincare_loss,
    'max_margin': max_margin_loss,
}


def register_loss_function(name: str, fn: Callable):
    _LOSS_MAPPING[name] = fn


@auto_register_save_load
class TorchLoss:
    """Find loss function from 'torch.nn.functional' and 'torch.nn'"""

    def __init__(self, loss_fn: str | Callable, **kwargs) -> None:
        # super().__init__()
        self.fn = None
        if isinstance(loss_fn, str):
            if loss_fn.lower() in _LOSS_MAPPING:
                self.fn = _LOSS_MAPPING[loss_fn.lower()]
            else:
                module = importlib.import_module('torch.nn.functional')
                fn = getattr(module, loss_fn, None)
                if fn is not None:
                    self.fn = lambda *arg, **kwd: fn(*arg, **kwd, **kwargs)
                else:
                    module = importlib.import_module('torch.nn')
                    t = getattr(module, loss_fn, None)
                    if t is not None:
                        self.fn = t(**kwargs)
                if self.fn is None:
                    raise RuntimeError(f'loss_fn {loss_fn} not found.')
        else:
            self.fn = loss_fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
