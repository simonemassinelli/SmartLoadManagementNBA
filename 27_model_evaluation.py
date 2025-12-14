import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_win_probs, all_win_targets = [], []
    all_injury_probs, all_injury_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            preds = model(batch)
            win_probs = torch.sigmoid(preds["win_logits"])
            win_targets = batch["won"].float()
            all_win_probs.append(win_probs.cpu())
            all_win_targets.append(win_targets.cpu())

            mask = batch["player_mask"].bool()
            injury_probs = torch.sigmoid(preds["injury_logits"])[mask]
            injury_targets = batch["injuries"].float()[mask]
            all_injury_probs.append(injury_probs.cpu())
            all_injury_targets.append(injury_targets.cpu())

    win_probs = torch.cat(all_win_probs).numpy().ravel()
    win_targets = torch.cat(all_win_targets).numpy().ravel()
    win_preds = (win_probs >= 0.5).astype(int)

    injury_probs = torch.cat(all_injury_probs).numpy()
    injury_targets = torch.cat(all_injury_targets).numpy()
    injury_preds = (injury_probs >= 0.5).astype(int)

    metrics = {
        "win_accuracy": accuracy_score(win_targets, win_preds),
        "win_auc": roc_auc_score(win_targets, win_probs),
        "injury_accuracy": accuracy_score(injury_targets, injury_preds),
        "injury_auc": roc_auc_score(injury_targets, injury_probs),
        "injury_precision": precision_score(injury_targets, injury_preds, zero_division=0),
        "injury_recall": recall_score(injury_targets, injury_preds, zero_division=0),
        "injury_f1": f1_score(injury_targets, injury_preds, zero_division=0),
    }

    return metrics, injury_targets, injury_preds

if __name__ == "__main__":
    from SmartLoadModel_19 import SmartLoadModel
    from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
    from dataloaders_22 import get_dataloaders

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

    metrics, injury_targets, injury_preds = evaluate_metrics(model, val_loader, device)

    cm = confusion_matrix(injury_targets, injury_preds)
    print("Confusion Matrix (Injuries):")
    print(cm)

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")