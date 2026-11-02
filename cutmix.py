'''CutMix (Yun et al., 2019): paste a random box from another image in the batch.'''
import math
import random

import numpy as np
import torch


def cutmix_batch(inputs, targets, alpha=1.0):
    '''Apply CutMix to a batch.

    Each image gets a box cut from a randomly chosen partner image in the same
    batch. The box area is drawn from Beta(alpha, alpha).

    Returns:
        mixed inputs, original targets, partner targets, and lam = fraction of
        each image that is still the original (used to weight the two losses).
    '''
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)

    H, W = inputs.shape[2], inputs.shape[3]
    cut_h, cut_w = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = random.randrange(H), random.randrange(W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)

    inputs = inputs.clone()
    inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    # The box may have been clipped at the border: recompute lam from the actual area.
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return inputs, targets, targets[index], lam
