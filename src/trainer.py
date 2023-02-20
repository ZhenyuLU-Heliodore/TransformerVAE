import torch
import torch.optim as optim
import numpy as np

from pathlib import Path

from .model import TransformerVAE
from .criterion import vae_clustering_loss, cluster_eval_metric


class Trainer:
    def __init__(self, args, train_loader, validation_loader):
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.result_path = args.result_path
        self.eval_metric = args.eval_metric

        self.model = TransformerVAE(
            num_classes=args.num_classes,
            num_tokens=args.num_tokens,
            pad_id=args.pad_id,
            seq_len=args.seq_len,
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
        self.model.to(args.device)
        self.device = args.device

        self.optimizer = optim.Adam(self.model.parameters(), args.lr)

    def train(self, epoch):
        losses, recon_losses, kl_losses, cat_losses = 0., 0., 0., 0.
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels, padding_mask = map(lambda x: x.to(self.device), batch)

            x_recon, z_mean, z_log_var, z_prior_mean, z, y = self.model(
                token_seq=inputs, key_padding_mask=padding_mask
            )

            loss, recon_loss, kl_loss, cat_loss = vae_clustering_loss(
                x_recon, inputs, z_log_var, z_prior_mean, y,
                lambda1=self.lambda1, lambda2=self.lambda2
            )

            losses += loss
            recon_losses += recon_loss
            kl_losses += kl_loss
            cat_losses += cat_loss

            self.optimizer.zero_grad()

            loss.backward()
            # recon_loss.backward()
            self.optimizer.step()

        training_result = (
            "Train Epoch: {}\t>\tLoss: {:.6f}\n recon_loss: {}, kl_loss: {}, cat_loss: {}"
            .format(
                epoch, losses / n_batches, recon_losses / n_batches,
                kl_losses / n_batches, cat_losses  / n_batches
            )
        )
        print(training_result)
        with open(self.result_path, 'a') as file:
            file.write(training_result + '\n')

    def validate(self, epoch):
        losses = 0
        n_batches = len(self.validation_loader)
        y_arrays, label_arrays = [], []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.validation_loader):
                inputs, labels, padding_mask = map(lambda x: x.to(self.device), batch)

                x_recon, z_mean, z_log_var, z_prior_mean, z, y = self.model(
                    token_seq=inputs, key_padding_mask=padding_mask
                )

                loss, _, _, _ = vae_clustering_loss(
                    x_recon, inputs, z_log_var, z_prior_mean, y,
                    lambda1=self.lambda1, lambda2=self.lambda2
                )
                # # another method to choose predicted y
                # y = z_prior_mean.square().mean(dim=-1).argmin(dim=1)

                y_arrays.append(y.cpu().numpy())
                label_arrays.append(labels.cpu().numpy())

                losses += loss

            score = eval_global_score(y_arrays, label_arrays, self.eval_metric)

        validation_result = (
            "Validation Epoch: {}\t>\tLoss: {:.6f}\tScore: {:.6f}"
            .format(epoch, losses / n_batches, score)
        )
        print(validation_result)
        with open(self.result_path, 'a') as file:
            file.write(validation_result + '\n')

    def save(self, epoch, model_prefix):
        path = Path(model_prefix + "_epoch_" + str(epoch) + ".pt")
        if not path.parent.exists():
            path.parent.mkdir()

        torch.save(self.model, path)


def eval_global_score(
        predicts: list, labels: list, eval_metric: str
):
    predicts_all = np.concatenate(predicts, axis=0)
    if predicts_all.shape.__len__() > 1:
        predicts_all = np.argmax(predicts_all, axis=-1)
    labels_all = np.concatenate(labels, axis=0)
    print(predicts_all, labels_all)
    score = cluster_eval_metric(predicts_all, labels_all, metric=eval_metric)

    return score