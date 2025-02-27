# src/loss.py
import torch
from torch import nn
    
class Loss(nn.Module):
    def __init__(self, lambda_reg):
        super(Loss, self).__init__()
        self.bce = nn.BCELoss(reduction='mean')
        self.sigm = nn.Sigmoid()
        self.lambda_reg = lambda_reg
    
    def forward(self, pred, target, batch_alphas):
        z_1o, z_1p, z_2o, z_2p = pred
        z_1o, z_1p, z_2o, z_2p = z_1o.squeeze(1), z_1p.squeeze(1), z_2o.squeeze(1), z_2p.squeeze(1)
        z_diff = z_1p - z_2p
        loss = 0

        ## Other
        p_other = 1 - self.sigm(-z_1o)*self.sigm(-z_2o)
        target_other = (target == 3).float()
        loss +=  self.bce(p_other, target_other)

        ## Stable, Better, Worse
        mask = (target != 3)
        batch_gammas = 2 ** batch_alphas
        p_diff = 1 / (1 + torch.exp(-z_diff[mask] * batch_gammas[mask]))

        target_no = torch.zeros_like(target[mask], dtype=torch.float)
        target_no[target[mask] == 0] = 1.0
        target_no[target[mask] == 1] = 0.5
        target_no[target[mask] == 2] = 0.0

        bce_loss = self.bce(p_diff, target_no)
        loss += bce_loss if not torch.isnan(bce_loss) else 0

        ## Regularization for gamma
        reg_loss = (self.lambda_reg * torch.abs(batch_alphas[mask])).mean()
        loss += reg_loss if not torch.isnan(reg_loss) else 0
        return loss