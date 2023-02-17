import torch
import numpy as np

from .trainer import eval_global_score


def inference(args, dataloader):
    model = torch.load(args.model_path).to(args.device)
    model.eval()

    with torch.no_grad():
        y_arrays, label_arrays = [], []
        for i, batch in enumerate(dataloader):
            inputs, labels, padding_mask = map(lambda x: x.to(args.device), batch)
            y = model(token_seq=inputs, key_padding_mask=padding_mask)[-1]

            y_arrays.append(y.cpu().numpy())
            label_arrays.append(labels.cpu().numpy())

        score = eval_global_score(y_arrays, label_arrays, args.eval_metric)

    predicts = np.concatenate(y_arrays, axis=0)
    labels = np.concatenate(label_arrays, axis=0)

    return score, predicts, labels