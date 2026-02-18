import numpy as np
import torch


def compute_nmse(x1, x2):
    x1 = x1 - torch.mean(x1, dim=0, keepdim=True)
    x1 = x1/torch.sum(x1**2)**0.5
    x2 = x2 - torch.mean(x2, dim=0, keepdim=True)
    x2 = x2/torch.sum(x2**2)**0.5

    nmse = torch.sum((x1-x2)**2)
    return nmse

def compute_nmse_np(x1, x2):
    x1 = x1 - np.mean(x1, axis=0, keepdims=True)
    x1 = x1/np.sum(x1**2)**0.5
    x2 = x2 - np.mean(x2, axis=0, keepdims=True)
    x2 = x2/np.sum(x2**2)**0.5

    nmse = np.sum((x1-x2)**2)
    return nmse
