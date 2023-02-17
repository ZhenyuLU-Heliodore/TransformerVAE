import argparse
import torch

from torch.utils.data import DataLoader
from src import Trainer


def train(args):
    print(args)
    with open(args.result_path, 'w') as file:
        file.write('args:\n' + str(args) + '\n')

    training_dataset = torch.load(args.training_set)
    validation_dataset = torch.load(args.validation_set)

    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.validation_batch_size, shuffle=True)

    trainer = Trainer(args, training_loader, validation_loader)

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.validate(epoch)
        trainer.save(epoch, args.model_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--encoder_dim", default=256, type=int)
    parser.add_argument("--decoder_dim", default=256, type=int)
    parser.add_argument("--num_encoder_layers", default=6, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--dim_ffn", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--layer_norm_eps", default=1e-5, type=float)
    parser.add_argument("--encoder_norm", default=None, type=torch.nn.Module)
    parser.add_argument("--decoder_norm", default=None, type=torch.nn.Module)
    parser.add_argument("--rpe", default=False, type=bool)

    # Dataset args
    parser.add_argument("--training_set", default="./dataset/training_set.pt", type=str)
    parser.add_argument("--validation_set", default="./dataset/validation_set.pt", type=str)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--num_tokens", default=256, type=int)
    parser.add_argument("--pad_id", default=256, type=int)
    parser.add_argument("--seq_len", default=256, type=int)

    # Trainer args
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--validation_batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--eval_metric", default="rand_score", type=str)
    parser.add_argument("--result_path", default="results.txt", type=str)
    parser.add_argument("--model_prefix", default="./models/TransformerVAE", type=str)

    # optimizer args
    parser.add_argument("--lambda1", default=1., type=float)
    parser.add_argument("--lambda2", default=3., type=float)
    parser.add_argument("--lr", default=1e-4, type=float)

    args = parser.parse_args()

    train(args)