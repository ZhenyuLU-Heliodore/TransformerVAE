import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    _gaussian_sampling,
    _get_activation_fn,
    _get_module_clones,
)


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            dim_ffn=512,
            dropout=0.1,
            activation="relu",
            layer_norm_eps=1e-5,
            rpe=False,
    ):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.rpe = rpe
        if rpe:
            pass

        self.linear1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        if not self.rpe:
            x = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )[0]
            return self.dropout1(x)
        else:
            pass

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            dim_ffn=512,
            dropout=0.1,
            activation="relu",
            layer_norm_eps=1e-5,
            rpe=False,
    ):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.rpe = rpe
        if rpe:
            pass

        self.linear1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        if not self.rpe:
            x = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )[0]
            return self.dropout1(x)
        else:
            pass

    def _mha_block(self, x, memory, attn_mask, key_padding_mask):
        x = self.multihead_attn(
            x, memory, memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout3(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_module_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer,
            num_layers,
            norm=None,
    ):
        super().__init__()
        self.layers = _get_module_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(
            self, tgt, memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        output=tgt
        for layer in self.layers:
            output = layer(
                output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class VAECorp(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_classes):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.mean = nn.Linear(encoder_dim, decoder_dim)
        self.log_var = nn.Linear(encoder_dim, decoder_dim)

        self.gaussian = GaussianPrior(num_classes, decoder_dim)
        self.classifier = Classifier(num_classes, decoder_dim)

    def forward(self, src):
        z_mean = F.relu(self.mean(src))
        z_log_var = F.relu(self.log_var(src))

        z = _gaussian_sampling(z_mean, z_log_var)
        z_prior_mean = self.gaussian(z)
        y = self.classifier(z)

        return z_mean, z_log_var, z, z_prior_mean, y


class GaussianPrior(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, latent_dim), requires_grad=True)

    def forward(self, z):
        z = z.unsqueeze(1)
        return z - self.mean.unsqueeze(0)


class Classifier(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        return F.softmax(self.linear2(h), dim=-1)