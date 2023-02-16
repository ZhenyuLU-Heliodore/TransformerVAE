import torch

from torch.nn import CrossEntropyLoss
from sklearn.metrics.cluster import adjusted_rand_score


def vae_clustering_loss(x_recon, x, log_var, z_prior_mean, y, lambda1=2., lambda2=3.):
    num_tokens = x_recon.size(dim=-1)
    cross_entropy = CrossEntropyLoss()

    # x_recon: [batch_size, seq_len-1, num_tokens] -> [-1, num_tokens]
    # x: [batch_size, num_tokens] -> [batch_size * (seq_len - 1)]
    # 257 classification without cls. pad_id 257 -> 256
    recon_loss = cross_entropy(
        x_recon.view(-1, num_tokens), x[:, 1:].reshape(-1).clamp(min=0, max=256)
    )
    kl_loss = -0.5 * (log_var.unsqueeze(1) - torch.square(z_prior_mean))
    kl_loss = torch.mean(torch.matmul(y.unsqueeze(1), kl_loss), 0)
    cat_loss = torch.mean(y * torch.log(y + 1e-10), 0)

    vae_loss = (
            lambda1 * torch.sum(recon_loss) + lambda2 * torch.sum(kl_loss) + torch.sum(cat_loss)
    )
    return vae_loss


def cluster_eval_metric(y, label, metric="adjusted_rand_score"):
    if metric == "adjusted_rand_score":
        return adjusted_rand_score(y, label)