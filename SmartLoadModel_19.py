import torch
import torch.nn as nn


from SelfAttentionPooling_17 import SelfAttentionPooling


class SmartLoadModel(nn.Module):
    def __init__(self, n_shared_features, n_player_features, n_win_features, n_injury_features, hidden_dim = 256, n_attention_heads = 4, dropout = 0.3, normalize_by_total = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.normalize_by_total = normalize_by_total

        # Player encoder
        self.player_encoder = nn.Sequential(
            nn.Linear(n_player_features + 1, hidden_dim), # + 1 for proposed minutes
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

        # shared context encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(n_shared_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Win prediction head (shared context + aggregated players + win features)
        self.win_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + n_win_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Binary - win/lose (1/0)
        )


        # Injury prediction head - per player, combines(player encoding + shared context + injury features)
        self.injury_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + n_injury_features + 1, hidden_dim), # + 1 for minutes
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch, proposed_minutes = None):
        '''
        Sample batch:
            shared_features     :torch.Size([32, 25])
            player_features     :torch.Size([32, 19, 36])
            actual_minutes      :torch.Size([32, 19])
            player_mask         :torch.Size([32, 19])
            win_features        :torch.Size([32, 20])
            injury_features     :torch.Size([32, 19, 29])
            won                 :torch.Size([32, 1])
            injuries            :torch.Size([32, 19])
            num_players         :torch.Size([32])

        proposed_minutes (32, 19) - 19 is max players, if None uses actual_minutes from batch (for training)

        Returns:
            win_logits: (32, 1) - win prob logits
            injury_logits: (32, 19) - injury prob logits per player
        '''

        B, max_players = batch['player_features'].shape[0], batch['player_features'].shape[1]
        mask = batch['player_mask']

        if proposed_minutes is None:
            proposed_minutes = batch['actual_minutes']
            # if we always will use actual minutes during training, the model can become too dependent on minutes, so we need some noise
            if self.training:
                proposed_minutes = self._augment_minutes(
                    proposed_minutes,
                    mask = batch['player_mask']
                )

        proposed_minutes = proposed_minutes.to(batch['player_features'].device).float()
        mask = mask.to(proposed_minutes.device).float()

        proposed_minutes = proposed_minutes * mask

        if self.normalize_by_total:
            team_total = proposed_minutes.sum(dim = 1, keepdim = True).clamp(min = 1.0)
            normalized_minutes = proposed_minutes / team_total
        else:
            normalized_minutes = (proposed_minutes / 48.0).clamp(0, 1)

        shared_encoded = self.shared_encoder(batch['shared_features']) # (32, 256)

        player_feats_with_minutes = torch.cat([
            batch['player_features'], # (32, 19, 36)
            normalized_minutes.unsqueeze(-1) # (32, 19, 1)
        ], dim = -1) # (32, 19, 36 + 1)


        player_encoded = self.player_encoder(player_feats_with_minutes) # (32, 19, 256)

        # Self attention + pooling (player interactions)
        attended_players, team_representation, attention_weights = self.self_attention_pooling(
            player_encoded,
            mask = batch['player_mask']
        )

        # Win predictions
        win_input = torch.cat([shared_encoded, team_representation, batch['win_features']], dim=-1)
        win_logits = self.win_head(win_input)



        # Injury predicitons
        shared_broadcast = shared_encoded.unsqueeze(1).expand(-1, max_players, -1)

        injury_input = torch.cat([
            attended_players,
            shared_broadcast,
            batch['injury_features'],
            normalized_minutes.unsqueeze(-1)
        ], dim = -1)

        injury_logits = self.injury_head(injury_input).squeeze(-1)

        predictions = {
            'win_logits': win_logits,
            'injury_logits': injury_logits,
            'attention_weights': attention_weights
        }

        return predictions



    def _augment_minutes(self, minutes, mask, total_minutes = 240, noise_std = 2.0):
        noise = torch.randn_like(minutes) * noise_std
        augmented = (minutes + noise).clamp(min = 0, max = 48)

        augmented = augmented * mask
        current_total = augmented.sum(dim = 1, keepdim = True).clamp(min = 1.0)
        augmented = augmented * (total_minutes / current_total)

        return augmented