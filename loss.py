import torch
import torch.nn as nn


class EveryStepLoss(nn.Module):

    def __init__(self, gamma):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma

    def forward(self, outputs, targets, lengths):
        loss = self.ce_loss(outputs, targets)
        exp_weights = torch.zeros(loss.size(0)).to(loss.device)
        beg = 0
        for i in range(len(lengths)):
            end = lengths[i] + beg
            exp_weights[beg:end] = torch.softmax(torch.linspace(
                -self.gamma, self.gamma, lengths[i]), dim=0)
            beg = end

        loss = torch.dot(loss, exp_weights) / lengths.size(0)

        return loss
