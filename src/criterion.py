import torch

from torch.nn import CrossEntropyLoss
from sklearn.metrics.cluster import rand_score, adjusted_rand_score


def vae_clustering_loss(x_recon, x, log_var, z_prior_mean, y, lambda1=1., lambda2=3.):
    num_tokens = x_recon.size(dim=-1) - 1 # counted excluding pad_id
    cross_entropy = CrossEntropyLoss()
    # x_recon: [batch_size, seq_len, num_tokens+1] -> [-1, num_tokens+1]
    # x: [batch_size, seq_len] -> [batch_size * seq_len]
    recon_loss = cross_entropy(
        x_recon.view(-1, num_tokens+1), x.view(-1).to(torch.int64)
    )
    kl_loss = -0.5 * (log_var.unsqueeze(1) - torch.square(z_prior_mean))
    kl_loss = torch.mean(torch.matmul(y.unsqueeze(1), kl_loss))

    cat_loss = torch.mean(y * torch.log(y + 1e-10))

    vae_loss = (
             lambda1 * recon_loss + lambda2 * kl_loss + cat_loss
    )
    return vae_loss, recon_loss, kl_loss, cat_loss


def cluster_eval_metric(predict, label, metric="rand_score"):
    if metric == "rand_score":
        return rand_score(label, predict)
    if metric == "adjusted_rand_score":
        return adjusted_rand_score(label, predict)
    else:
        raise ValueError(
            "metric is allowed in {} and {}, but got {}".format(
                "rand_score", "adjusted_rand_score", metric
            )
        )