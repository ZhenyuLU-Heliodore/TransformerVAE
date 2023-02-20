import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def _get_module_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, but got {}".format(activation))


def _gaussian_sampling(mean, log_var):
    std = torch.exp(0.5 * log_var)
    noise = torch.randn_like(std)

    return noise.mul(std).add(mean)


def _get_sinusoidal_pe(seq_len, dim, n=10000):
    pe = np.zeros((seq_len, dim), dtype=float)
    for p in range(seq_len):
        for i in np.arange(int(dim / 2)):
            denominator = np.power(n, 2 * i / dim)
            pe[p, 2 * i] = np.sin((p / denominator))
            pe[p, 2 * i + 1] = np.cos(p / denominator)
    return pe


def _get_target_mask(seq_len):
    attn_shape = (seq_len, seq_len)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask != 0