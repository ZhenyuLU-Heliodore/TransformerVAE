import argparse
import torch

from src import clustering_visualization, inference
from torch.utils.data import DataLoader


def evaluate(args):
    print(args)

    dataset = torch.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.evaluate_batch_size, shuffle=True)

    score, predicts, labels = inference(args, dataloader)

    print(score)
    clustering_visualization(args, predicts, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path args
    parser.add_argument("--dataset_path", default="./dataset/validation_set.pt", type=str)
    parser.add_argument("--model_path", default="./models/TransformerVAE_epoch_1.pt", type=str)
    parser.add_argument("--figure_prefix", default="./figures/clustering", type=str)

    # evaluation args
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--eval_metric", default="rand_score", type=str)
    parser.add_argument("--evaluate_batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    # plotting args
    parser.add_argument("--max_num_points", default=50, type=int)
    parser.add_argument("--perplexity", default=30., type=float)

    args = parser.parse_args()

    evaluate(args)