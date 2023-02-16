import argparse
import torch

from torch.utils.data import DataLoader
from src import Trainer


def train(
        args, result_path="results.txt",
        training_set="./dataset/training_set.pt",
        validation_set="./dataset/validation_set.pt",
):
    print(args)
    with open(result_path, 'w') as file:
        file.write('args:\n' + str(args) + '\n')

    training_dataset = torch.load(training_set)
    validation_dataset = torch.load(validation_set)
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    trainer = Trainer(
        args, training_loader, validation_loader,
        num_classes=4, result_path=result_path, device=args.device,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.validate(epoch)
        trainer.save(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lambda1", default=2., type=float)
    parser.add_argument("--lambda2", default=3., type=float)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--encoder_dim", default=256, type=int)
    parser.add_argument("--decoder_dim", default=256, type=int)
    parser.add_argument("--num_encoder_layers", default=6, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--dim_ffn", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--layer_norm_eps", default=1e-5, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--encoder_norm", default=None, type=torch.nn.Module)
    parser.add_argument("--decoder_norm", default=None, type=torch.nn.Module)
    parser.add_argument("--rpe", default=False, type=bool)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epochs", default=10, type=int)

    args = parser.parse_args()

    train(args)