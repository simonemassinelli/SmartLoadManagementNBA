import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads = 4, dropout = 0.3):
        super(SelfAttentionPooling, self).__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, player_encodings, mask = None):
        # mask: 1 for real players, 0 for padding
        attn_mask = (mask == 0) if mask is not None else None

        attended, attention_weights = self.attention(
            query = player_encodings,
            key = player_encodings,
            value = player_encodings,
            key_padding_mask = attn_mask
        )

        attended = self.dropout(attended)
        attended_output = self.norm(player_encodings + attended)

        # masked aggregation
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_output = attended_output * mask_expanded

            num_active = mask.sum(dim = 1, keepdim = True).clamp(min = 1)
            aggregated = masked_output.sum(dim = 1) / num_active
        else:
            aggregated = attended_output.mean(dim = 1)

        return attended_output, aggregated, attention_weights

