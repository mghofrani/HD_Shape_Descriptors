import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetRegressionLoss(nn.Module):
    def __init__(self, reg_weight=0, size_average=True):
        super(PointNetRegressionLoss, self).__init__()
        self.reg_weight = reg_weight
        self.size_average = size_average

    def forward(self, predictions, targets, A=None):
        # Ensure that targets have the same shape as predictions
        targets = targets.view(-1, 1)

        # Compute regression loss (e.g., Mean Squared Error)
        regression_loss = F.mse_loss(predictions, targets, reduction='mean')

        # Compute regularization term if needed
        reg = 0
        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1)
            if A.is_cuda:
                I = I.cuda()
            reg = torch.norm(I - torch.bmm(A, A.transpose(2, 1)), p='fro')  # Frobenius norm
            reg = self.reg_weight * reg / predictions.size(0)  # Normalize by batch size

        # Compute total loss
        loss = regression_loss + reg

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class AutoEncoderLoss(nn.Module):
    def __init__(self):
        super(AutoEncoderLoss, self).__init__()
            
    def forward(self, predictions, targets):
        chamfer_loss = ChamferLoss()
        # Calculate Chamfer Distance or Earth Mover's Distance based on user preference
        distance = chamfer_loss(predictions, targets)
        return distance
