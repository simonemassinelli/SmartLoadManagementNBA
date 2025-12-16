import torch
import math

from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES

from dataloaders_22 import get_dataloaders

def count_params(model):
    total = sum(param.numel() for param in model.parameters())
    return total

def get_len_features(features):
    return len(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\nba_game_features_final.csv"
train_loader, val_loader, test_loader = get_dataloaders(path, 32, 0.8, 0.1)

batch = next(iter(train_loader))
print(f"Sample batch:")

for key, value in batch.items():
    if hasattr(value, 'shape'):
        print(f"{key:20s}:{value.shape}")


model = SmartLoadModel(
    n_shared_features = get_len_features(SHARED_FEATURES),
    n_player_features = get_len_features(PLAYER_FEATURES),
    n_win_features = get_len_features(WIN_FEATURES),
    n_injury_features = get_len_features(INJURY_FEATURES),
    hidden_dim = 256,
    n_attention_heads = 4,
    dropout = 0.3
).to(device)

total = count_params(model)
print(model)
print(f"Total params: {total:,}")


batch = next(iter(train_loader))

for key, value in batch.items():
    if torch.is_tensor(value):
        batch[key] = value.to(device)

model.eval()
with torch.no_grad():
    predictions = model(batch)

print("win_logits:", predictions["win_logits"].shape)
print("injury_logits:", predictions["injury_logits"].shape)
print("attn:", predictions["attention_weights"].shape)


mask = batch["player_mask"][0]  # (19,)
print("mask:", mask)
print("num real:", int(mask.sum().item()))

attn =  predictions["attention_weights"]
attn0 = attn[0]  # (19,19)
real = int(mask.sum().item())
print("Attn real->pad sum:", attn0[:real, real:].sum().item())
print("Attn pad->real sum:", attn0[real:, :real].sum().item())


def check_tensor(name, t):
    print(f"{name:15s}  min={t.min().item(): .3f}  max={t.max().item(): .3f}  nan={torch.isnan(t).any().item()}")

check_tensor("win_logits", predictions["win_logits"])
check_tensor("injury_logits", predictions["injury_logits"])