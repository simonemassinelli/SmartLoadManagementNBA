import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartLoadLoss(nn.Module):
    def __init__(self, win_weight = 1.0, injury_weight = 1.0):
        super().__init__()

        self.win_weight = win_weight
        self.injury_weight = injury_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, player_mask):
        # Win loss
        win_loss = self.bce_loss(
            predictions['win_logits'],
            targets['won']
        )

        injury_logits = predictions['injury_logits']
        injury_targets = targets['injuries']
        mask = player_mask.float()

        per_player_loss = F.binary_cross_entropy_with_logits(
            injury_logits,
            injury_targets,
            reduction='none'
        )

        # mask and average (only over active players)
        injury_loss = (per_player_loss * mask).sum() / mask.sum().clamp(min = 1.0)

        total_loss = self.win_weight * win_loss + self.injury_weight * injury_loss

        return total_loss, win_loss, injury_loss