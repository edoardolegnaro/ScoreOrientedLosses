import torch

pi = torch.pi

def F_unif(x):
    return x


def F_cos(x, mu, delta):
    return torch.where(
        x < mu - delta,
        torch.zeros_like(x),
        torch.where(
            x > mu + delta,
            torch.ones_like(x),
            0.5 * (1 + (x - mu) / delta + (1 / pi) * torch.sin(pi * (x - mu) / delta)),
        ),
    )

def pdf_F_cos(x, mu, delta):
    return 1 / (2 * delta) * (1 + torch.cos(pi * (x - mu) / delta))
