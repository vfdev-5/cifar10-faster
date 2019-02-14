
import random
import numpy as np
import torch


def mixup_data(x, y, alpha, proba):    
    lam = np.random.beta(alpha, alpha) if alpha > 0 and random.random() < proba else 1.0
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, (y, y[index], lam)

class MixupCriterion:

    def __init__(self, base_criterion):
        self.criterion = base_criterion

    def __call__(self, y_pred, y):        
        y_a, y_b, lam = y
        return lam * self.criterion(y_pred, y_a) + (1 - lam) * self.criterion(y_pred, y_b)

    def __getattr__(self, attr):        
        def wrapper(*args, **kwargs):
            return getattr(self.criterion, attr)(*args, **kwargs)
        return wrapper
