from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
from dataloaders_22 import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\nba_game_features_final.csv"

train_loader, val_loader, test_loader = get_dataloaders(path, batch_size=32, train_split=0.8, val_split=0.1)

model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
    hidden_dim=256,
    n_attention_heads=4,
    dropout=0.3
).to(device)

win_loss_fn = nn.BCEWithLogitsLoss()
injury_loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 5

for epoch in range(n_epochs):
    model.train()
    total_win_loss = 0.0
    total_injury_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        preds = model(batch)
        win_logits = preds["win_logits"]
        injury_logits = preds["injury_logits"]
        win_target = batch["won"].float()
        injury_target = batch["injuries"].float()
        win_loss = win_loss_fn(win_logits, win_target)
        injury_loss = injury_loss_fn(injury_logits, injury_target)
        loss = win_loss + injury_loss
        loss.backward()
        optimizer.step()
        total_win_loss += win_loss.item() * win_target.size(0)
        total_injury_loss += injury_loss.item() * win_target.size(0)

    n_samples = len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Win loss={total_win_loss/n_samples:.4f}, Injury loss={total_injury_loss/n_samples:.4f}")

    model.eval()
    val_win_loss = 0.0
    val_injury_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            preds = model(batch)
            win_logits = preds["win_logits"]
            injury_logits = preds["injury_logits"]
            win_target = batch["won"].float()
            injury_target = batch["injuries"].float()
            val_win_loss += win_loss_fn(win_logits, win_target).item() * win_target.size(0)
            val_injury_loss += injury_loss_fn(injury_logits, injury_target).item() * win_target.size(0)

    print(f"Validation: Win loss={val_win_loss/n_samples:.4f}, Injury loss={val_injury_loss/n_samples:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
}, r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\smartload_model.pt")