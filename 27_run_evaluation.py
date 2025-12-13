import torch

from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
from dataloaders_22 import get_dataloaders
from model_evaluation_26 import evaluate_model, injury_classification_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "nba_game_features_final.csv"

train_loader, val_loader, test_loader = get_dataloaders(
    path,
    batch_size=32,
    train_split=0.8,
    val_split=0.1
)

model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
).to(device)

model.load_state_dict(torch.load("model.pt", map_location=device))

metrics = evaluate_model(model, val_loader, device)
print(metrics)

precision, recall, f1 = injury_classification_metrics(model, val_loader, device)
print(f"Injury Precision: {precision:.3f}")
print(f"Injury Recall:    {recall:.3f}")
print(f"Injury F1:        {f1:.3f}")