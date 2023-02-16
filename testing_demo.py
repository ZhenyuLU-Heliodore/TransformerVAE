import torch

from src.model import TransformerVAE
from torch.nn import CrossEntropyLoss


if __name__ == "__main__":
    num_classes , batch_size, seq_len, dim = 4, 16, 257, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_sequence = torch.stack([torch.arange(seq_len)] * batch_size, dim=0).to(device)
    key_padding_mask = torch.cat(
        (torch.zeros((batch_size, 100)), torch.ones(batch_size, seq_len - 100)), dim=1
    ).to(device)

    model = TransformerVAE(num_classes=num_classes).to(device)
    x_recon, z_mean, z_log_var, z_prior_mean, z, y = model(
        token_seq=token_sequence, key_padding_mask=key_padding_mask
    )

    # print("token_sequence: {}".format(token_sequence.shape))
    # print("key_padding_mask: {}".format(key_padding_mask.shape))
    # print(
    #     "x_recon: {}\nz_mean: {}\nz_log_var: {}\nz_prior_mean: {}\nz: {}\ny: {}".format(
    #         x_recon.shape, z_mean.shape, z_log_var.shape, z_prior_mean.shape, z.shape, y.shape
    #     )
    # )

    assert x_recon.shape == (batch_size, seq_len - 1, num_classes)
    assert z_mean.shape == (batch_size, dim)
    assert z_log_var.shape == (batch_size, dim)
    assert z_prior_mean.shape == (batch_size, num_classes, dim)
    assert z.shape == (batch_size, dim)
    assert y.shape == (batch_size, num_classes)
