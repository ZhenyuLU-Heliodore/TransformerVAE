import torch
import torch.nn as nn

from .modules import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    VAECorp,
    Classifier,
)

from .utils import (
    _get_sinusoidal_pe,
    _get_target_mask,
)


class TransformerVAE(nn.Module):
    def __init__(
            self,
            num_classes,
            num_tokens=256,
            pad_id=257,
            seq_len=257, # with cls as the 0th token
            num_heads=8,
            encoder_dim=256,
            decoder_dim=256,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_ffn=512,
            dropout=0.1,
            activation="relu",
            layer_norm_eps=1e-5,
            rpe=False,
            encoder_norm=None,
            decoder_norm=None,
    ):
        super().__init__()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=encoder_dim,
            num_heads=num_heads,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            rpe=rpe,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )
        self.decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            num_heads=decoder_dim,
            dim_ffn=dim_ffn,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            rpe=rpe,
        )
        self.decoder = TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm,
        )
        # self.decoder_layer = nn.TransformerDecoderLayer(decoder_dim, num_heads, dim_feedforward=dim_ffn, batch_first=True)
        # self.decoder = TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6)

        self.token_embedding = nn.Embedding(num_tokens + 2, encoder_dim, padding_idx=pad_id)
        self.vae_corp = VAECorp(encoder_dim, decoder_dim, num_classes)
        self.z_to_memory = nn.Linear(1, seq_len - 1)
        self.recon_classifier = Classifier(num_tokens + 1, decoder_dim)

        self.sinusoidal_pe = _get_sinusoidal_pe(seq_len, d=encoder_dim)
        self.seq_len = seq_len

    def forward(self, token_seq, key_padding_mask=None, mask=None):
        # [b, l] -> [b, l, d1], b = batch_size, l = seq_len, d1 = encoder_dim
        src = (
                self.token_embedding(token_seq) +
                torch.tensor(
                    self.sinusoidal_pe,
                    device=token_seq.device,
                    dtype=token_seq.dtype
                )
        )
        src = self.encoder(
            src, mask=mask, src_key_padding_mask=key_padding_mask
        )

        # [b, l, d1] -> [b, d1]
        cls = src[:, 0, :]
        # z_mean, z_log_var, z: [b, d2], d2 = decoder_dim
        # z_prior_mean: [b, c, d2], c = num_classes
        # y: [b, c]
        z_mean, z_log_var, z, z_prior_mean, y = self.vae_corp(cls)
        # [b, d2] -> [b, l-1, d2], excluding cls
        memory = self.z_to_memory(torch.unsqueeze(z, dim=2)).permute(0, 2, 1)
        tgt = torch.zeros_like(memory, requires_grad=True)
        tgt_mask = _get_target_mask(self.seq_len - 1).to(tgt.device)

        # [b, l-1, d2] -> [b, l-1, c]
        x_recon = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask[:, 1:]
        )
        x_recon = self.recon_classifier(x_recon)
        return x_recon, z_mean, z_log_var, z_prior_mean, z, y
