import torch
import torch.nn as nn
import torch.optim as optim
from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
    hidden_dim=256,
    n_attention_heads=4,
    dropout=0.3
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

checkpoint_path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\smartload_model.pt"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"] + 1

print(f"Checkpoint caricato, riprendo dal epoch {start_epoch}")