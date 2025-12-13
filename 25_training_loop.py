import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
from dataloaders_22 import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\nba_game_features_final.csv"
train_loader, val_loader, test_loader = get_dataloaders(path, batch_size=32, train_frac=0.8, val_frac=0.1)

model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
    hidden_dim=256,
    n_attention_heads=4,
    dropout=0.3
).to(device)

# Ottimizzatore e funzioni di loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn_win = nn.BCEWithLogitsLoss()       # logit -> probabilità
loss_fn_injury = nn.BCEWithLogitsLoss()

n_epochs = 10

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        # Sposta tutto su device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        preds = model(batch)

        # Target
        win_target = batch['win_target'].float()  # shape: (B, 1)
        injury_target = batch['injury_target'].float()  # shape: (B, P)

        # Calcola le loss
        loss_win = loss_fn_win(preds['win_logits'], win_target)
        loss_injury = loss_fn_injury(preds['injury_logits'], injury_target)

        # Combina le loss (qui semplicemente somma)
        loss = loss_win + loss_injury
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * win_target.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Avg loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            preds = model(batch)
            win_target = batch['win_target'].float()
            injury_target = batch['injury_target'].float()
            loss_win = loss_fn_win(preds['win_logits'], win_target)
            loss_injury = loss_fn_injury(preds['injury_logits'], injury_target)
            loss = loss_win + loss_injury
            val_loss += loss.item() * win_target.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch+1}, Validation loss: {avg_val_loss:.4f}")