from torch import nn


l1_loss = nn.L1Loss(reduction='sum')
l2_loss = nn.MSELoss(reduction='sum')
