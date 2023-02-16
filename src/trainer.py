import torch
import torch.optim as optim

from pathlib import Path

from .model import TransformerVAE
from .criterion import vae_clustering_loss


class Trainer:
    def __init__(
            self, args, train_loader, validation_loader, num_classes,
            num_tokens=256, pad_id=257, seq_len=257, device=None, result_path="results.txt",
    ):
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.result_path = result_path

        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        self.model = TransformerVAE(
            num_classes=num_classes,
            num_tokens=num_tokens,
            pad_id=pad_id,
            seq_len=seq_len,
            num_heads=args.num_heads,
            encoder_dim=args.encoder_dim,
            decoder_dim=args.decoder_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_ffn=args.dim_ffn,
            dropout=args.dropout,
            activation=args.activation,
            layer_norm_eps=args.layer_norm_eps,
            rpe=args.rpe,
            encoder_norm=args.encoder_norm,
            decoder_norm=args.decoder_norm,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), args.lr)

    def train(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels, padding_mask = map(lambda x: x.to(self.device), batch)

            x_recon, z_mean, z_log_var, z_prior_mean, z, y = self.model(
                token_seq=inputs, key_padding_mask=padding_mask
            )

            loss = vae_clustering_loss(
                x_recon, inputs, z_log_var, z_prior_mean, y,
                lambda1=self.lambda1, lambda2=self.lambda2
            )
            losses += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        training_result = 'Train Epoch: {}\t>\tLoss: {:.4f}'.format(epoch, losses / n_batches)
        print(training_result)
        with open(self.result_path, 'a') as file:
            file.write(training_result + '\n')

    def validate(self, epoch):
        losses = 0
        n_batches = len(self.train_loader)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.validation_loader):
                inputs, labels, padding_mask = map(lambda x: x.to(self.device), batch)
                x_recon, z_mean, z_log_var, z_prior_mean, z, y = self.model(
                    token_seq=inputs, key_padding_mask=padding_mask
                )

                loss = vae_clustering_loss(
                    x_recon, inputs, z_log_var, z_prior_mean, y,
                    lambda1=self.lambda1, lambda2=self.lambda2
                )
                losses += loss

        validation_result = "Validation Epoch: {}\t>\tLoss: {:.4f}".format(epoch, losses / n_batches)
        print(validation_result)
        with open(self.result_path, 'a') as file:
            file.write(validation_result + '\n')

    def save(self, epoch, model_prefix='model', root='model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()

        torch.save(self.model, path)