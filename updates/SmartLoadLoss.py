import torch
import torch.nn as nn
import torch.nn.functional as F


class SmartLoadLoss(nn.Module):
    def __init__(
        self,
        win_weight=1.0,
        injury_weight=1.0,
        pos_weight_value=30.0,
        label_smoothing=0.0
    ):
        super().__init__()
        self.win_weight = win_weight
        self.injury_weight = injury_weight
        self.pos_weight_value = pos_weight_value
        self.label_smoothing = label_smoothing

    def forward(self, predictions, targets, player_mask):
        win_loss = self._compute_win_loss(predictions, targets)
        injury_loss, injury_stats = self._compute_injury_loss(
            predictions, targets, player_mask
        )

        total_loss = self.win_weight * win_loss + self.injury_weight * injury_loss

        return total_loss, win_loss, injury_loss, injury_stats

    def _compute_win_loss(self, predictions, targets):
        win_logits = predictions["win_logits"]
        win_targets = targets["won"].float()

        if self.label_smoothing > 0:
            win_targets = win_targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(win_logits, win_targets)

    def _compute_injury_loss(self, predictions, targets, player_mask):
        injury_logits = predictions["injury_logits"]
        injury_targets = targets["injuries"].float()
        mask = player_mask.float()

        pos_weight = torch.tensor(
            [self.pos_weight_value],
            device=injury_logits.device,
            dtype=injury_logits.dtype
        )

        per_player_loss = F.binary_cross_entropy_with_logits(
            injury_logits, injury_targets, pos_weight=pos_weight, reduction="none"
        )

        masked_loss = per_player_loss * mask
        num_players = mask.sum().clamp(min=1.0)
        injury_loss = masked_loss.sum() / num_players

        with torch.no_grad():
            injury_probs = torch.sigmoid(injury_logits)
            injury_preds = (injury_probs > 0.5).float()

            active_targets = injury_targets * mask
            active_preds = injury_preds * mask

            true_positives = ((active_preds == 1) & (active_targets == 1)).sum()
            predicted_positives = (active_preds == 1).sum().clamp(min=1)
            actual_positives = (active_targets == 1).sum().clamp(min=1)

            injury_stats = {
                "injury_rate": (actual_positives / num_players).item(),
                "injury_precision": (true_positives / predicted_positives).item(),
                "injury_recall": (true_positives / actual_positives).item(),
                "injury_pred_rate": (predicted_positives / num_players).item()
            }

        return injury_loss, injury_stats

    @staticmethod
    def calculate_pos_weight(dataset, sample_size=1000):
        total_injuries = 0
        total_players = 0

        indices = torch.randperm(len(dataset))[:sample_size]

        for idx in indices:
            batch = dataset[idx.item()]
            mask = batch["player_mask"]
            injuries = batch["injuries"]

            total_injuries += (injuries * mask).sum().item()
            total_players += mask.sum().item()

        injury_rate = total_injuries / max(total_players, 1)
        pos_weight = (1 - injury_rate) / max(injury_rate, 1e-6)

        print(f"Injury rate: {injury_rate:.4f} ({injury_rate*100:.2f}%)")
        print(f"Recommended pos_weight: {pos_weight:.1f}")

        return pos_weight