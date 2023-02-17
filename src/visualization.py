import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.manifold import TSNE


def clustering_visualization(args, predicts: np.ndarray, labels: np.ndarray):
    num_classes = args.num_classes
    max_num_points = args.max_num_points

    predicts_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=args.perplexity
    ).fit_transform(predicts)

    if predicts.__len__() > max_num_points:
        points = predicts_embedded[0: max_num_points, :]
        colors = labels[0: max_num_points].astype(float)
    else:
        points = predicts_embedded
        colors = labels.astype(float)

    print(points[:, 0])
    print(points[:, 1])
    print(colors)

    if num_classes > 8:
        plt.scatter(points[:, 0], points[0:, 1])
    else:
        plt.scatter(points[:, 0], points[:, 1], c=colors * 10, alpha=0.5, cmap="viridis")

    path = Path(args.figure_prefix + ".png")
    if not path.parent.exists():
        path.parent.mkdir()
    plt.savefig(path)