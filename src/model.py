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
            pad_id=256,
            seq_len=256,
            num_heads=8,
            encoder_dim=256,
            decoder_dim=256,
            num_encoder_layers=6,
            num_decoder_layers=6,
            max_batch_size=128,
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

        #class token
        self.cls_embedding_table = nn.Embedding(max_batch_size, encoder_dim)
        self.token_embedding_table = nn.Embedding(num_tokens + 1, encoder_dim, padding_idx=pad_id)

        self.vae_corp = VAECorp(encoder_dim, decoder_dim, num_classes)
        self.z_to_memory = nn.Linear(1, seq_len)
        self.recon_classifier = Classifier(decoder_dim, num_tokens + 1)

        self.sinusoidal_pe = _get_sinusoidal_pe(seq_len, encoder_dim)
        self.target_mask = _get_target_mask(seq_len)

        self.seq_len = seq_len

    def forward(self, token_seq, key_padding_mask=None, mask=None):
        device = token_seq.device
        batch_size = token_seq.size(dim=0)

        # Embedding and encoding part

        # [b, 1, d1], b = batch_size, d1 = encoder_dim
        cls_embedding = self.cls_embedding_table.weight[0: batch_size, :].unsqueeze(1)
        # [l, d1], l = seq_len
        positional_embedding = torch.tensor(
            self.sinusoidal_pe, device=device, dtype=token_seq.dtype
        )
        # [b, l, d1]
        token_embedding = self.token_embedding_table(token_seq)

        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        # [b, l+1, d1]
        src = torch.cat((token_embedding + positional_embedding, cls_embedding), dim=1)
        # [b, l+1]
        padding_mask = torch.cat((key_padding_mask, cls_mask), dim=-1)

        src = self.encoder(src, mask=mask, src_key_padding_mask=padding_mask)

        # VAE part

        # [b, d1]
        cls = src[:, 0, :].squeeze(1)
        # z_mean, z_log_var, z: [b, d2], d2 = decoder_dim
        # z_prior_mean: [b, c, d2], c = num_classes
        # y: [b, c]
        z_mean, z_log_var, z, z_prior_mean, y = self.vae_corp(cls)

        # [b, d2] -> [b, l, d2]
        memory = self.z_to_memory(torch.unsqueeze(z, dim=-1)).permute(0, 2, 1)

        # Decoding part

        tgt = torch.zeros_like(memory, requires_grad=True)
        tgt_mask = torch.tensor(self.target_mask, dtype=torch.bool, device=device)

        x_recon = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask
        )
        # [b, l, d2] -> [b, l, c]
        x_recon = self.recon_classifier(x_recon)

        return x_recon, z_mean, z_log_var, z_prior_mean, z, y
