import torch
import torch.nn as nn
from SelfAttentionPooling_17 import SelfAttentionPooling

class SmartLoadModel(nn.Module):
    def __init__(self, n_shared_features, n_player_features, n_win_features, n_injury_features,
                 hidden_dim=256, n_attention_heads=4, dropout=0.3, normalize_by_total=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.normalize_by_total = normalize_by_total

        player_input_dim = n_player_features * 2 + 2

        self.player_encoder = nn.Sequential(
            nn.Linear(player_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.self_attention_pooling = SelfAttentionPooling(
            input_dim=hidden_dim,
            num_heads=n_attention_heads,
            dropout=dropout
        )

        self.shared_encoder = nn.Sequential(
            nn.Linear(n_shared_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.win_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + n_win_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.injury_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + n_injury_features + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch, proposed_minutes=None):
        B, max_players = batch['player_features'].shape[0], batch['player_features'].shape[1]
        mask = batch['player_mask']

        if proposed_minutes is None:
            proposed_minutes = batch['actual_minutes']

        proposed_minutes = proposed_minutes.to(batch['player_features'].device).float()
        mask = mask.to(proposed_minutes.device).float()
        proposed_minutes = proposed_minutes * mask

        if self.normalize_by_total:
            team_total = proposed_minutes.sum(dim=1, keepdim=True).clamp(min=1.0)
            normalized_minutes = proposed_minutes / team_total
        else:
            normalized_minutes = proposed_minutes / 48.0

        minutes_gate = normalized_minutes.unsqueeze(-1)
        gated_features = batch['player_features'] * minutes_gate

        player_feats_with_minutes = torch.cat([
            batch['player_features'],
            gated_features,
            normalized_minutes.unsqueeze(-1),
            (normalized_minutes ** 2).unsqueeze(-1)
        ], dim=-1)

        shared_encoded = self.shared_encoder(batch['shared_features'])
        player_encoded = self.player_encoder(player_feats_with_minutes)

        attended_players, team_representation, attention_weights = self.self_attention_pooling(
            player_encoded,
            mask=batch['player_mask']
        )

        win_input = torch.cat([shared_encoded, team_representation, batch['win_features']], dim=-1)
        win_logits = self.win_head(win_input)

        shared_broadcast = shared_encoded.unsqueeze(1).expand(-1, max_players, -1)

        injury_input = torch.cat([
            attended_players,
            shared_broadcast,
            batch['injury_features'],
            normalized_minutes.unsqueeze(-1),
            (normalized_minutes ** 2).unsqueeze(-1)
        ], dim=-1)

        injury_logits = self.injury_head(injury_input).squeeze(-1)

        return {
            'win_logits': win_logits,
            'injury_logits': injury_logits,
            'attention_weights': attention_weights
        }