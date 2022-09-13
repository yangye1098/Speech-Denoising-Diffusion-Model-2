from torch import nn


l1_loss = nn.L1Loss(reduction='mean')
l2_loss = nn.MSELoss(reduction='mean')
def log_loss(pred, target):
    return (pred - target).abs().mean(dim=-1).clamp(min=1e-20).log().mean()
