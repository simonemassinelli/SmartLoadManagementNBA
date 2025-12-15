from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
import wandb

from updates.SmartLoadModel import SmartLoadModel
from updates.features import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
from dataloaders_22 import get_dataloaders
from updates.SmartLoadLoss import SmartLoadLoss
from updates.model_evaluation import (
    evaluate_metrics,
    find_optimal_threshold,
    find_threshold_at_recall,
    evaluate_risk_tiers
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = Path('../../checkpoints')
checkpoint_dir.mkdir(exist_ok = True)


config = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 30,
    'hidden_dim': 256,
    'n_attention_heads': 4,
    'dropout': 0.2, # 0.3
    'win_weight': 1.0,
    'injury_weight': 0.5,
    'train_split': 0.8,
    'val_split': 0.1,
    'optimizer': 'AdamW',
    'scheduler': 'ReduceLROnPlateau',
    'early_stopping_patience': 5,
    'n_shared_features': len(SHARED_FEATURES),
    'n_player_features': len(PLAYER_FEATURES),
    'n_win_features': len(WIN_FEATURES),
    'n_injury_features': len(INJURY_FEATURES),
}

wandb.login(key='6abdf6894594565ea82f3de1776c314263de1089')

wandb.init(
    project="nba_model",
    name="baseline-run",
    config=config,
    tags=["baseline", "attention", "early-stopping", "lr-scheduler"]
)


path = r"../data/nba_game_features_final.csv"
train_loader, val_loader, test_loader = get_dataloaders(
    csv_path=path,
    batch_size=config['batch_size'],
    train_split=config['train_split'],
    val_split=config['val_split']
)



model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
    hidden_dim=256,
    n_attention_heads=4,
    dropout=0.3
).to(device)
wandb.watch(model, log='all', log_freq=100)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

train_dataset = train_loader.dataset
if hasattr(train_dataset, 'dataset'):
    base_dataset = train_dataset.dataset
else:
    base_dataset = train_dataset

optimal_pos_weight = SmartLoadLoss.calculate_pos_weight(base_dataset, sample_size = 2000)

loss_fn = SmartLoadLoss(
    win_weight=1.0,
    injury_weight=1.0,
    pos_weight_value=30.0,
    label_smoothing=0.01
)

n_epochs = config['epochs']
best_val_loss = float('inf')

patience = config['early_stopping_patience']
epochs_no_improve = 0
best_epoch = 0

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    total_win_loss = 0.0
    total_injury_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        # Sposta tutto su device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        preds = model(batch)

        # Target
        targets = {
            'won' : batch['won'],
            'injuries' : batch['injuries']
        }

        loss, win_loss, injury_loss, injury_stats = loss_fn(preds, targets, batch["player_mask"])


        loss.backward()
        optimizer.step()

        batch_size = batch['won'].size(0)
        total_loss += loss.item() * batch_size
        total_win_loss += win_loss.item() * batch_size
        total_injury_loss += injury_loss.item() * batch_size

    # Calcola le loss
    avg_loss = total_loss / len(train_loader.dataset)
    avg_win_loss = total_win_loss / len(train_loader.dataset)
    avg_injury_loss = total_injury_loss / len(train_loader.dataset)

    print(f"\nEpoch {epoch + 1} / {n_epochs}")
    print(f"Train - Total: {avg_loss:.4f} | Win: {avg_win_loss:.4f} | Injury: {avg_injury_loss:.4f}")

    model.eval()
    val_loss = 0.0
    val_win_loss = 0.0
    val_injury_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            preds = model(batch)

            targets = {
                'won' : batch['won'],
                'injuries' : batch['injuries']
            }
            loss, win_loss, injury_loss, injury_stats = loss_fn(preds, targets, batch["player_mask"])

            batch_size = batch['won'].size(0)
            val_loss += loss.item() *  batch_size
            val_win_loss += win_loss.item() * batch_size
            val_injury_loss += injury_loss.item() * batch_size

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_win_loss = val_win_loss / len(val_loader.dataset)
    avg_val_injury_loss = val_injury_loss / len(val_loader.dataset)

    print(f"Val - Total: {avg_val_loss:.4f} | Win: {avg_val_win_loss:.4f} | Injury: {avg_val_injury_loss:.4f}")

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"LR: {current_lr:.6f}")

    gap = avg_loss - avg_val_loss
    print(f"Train-Val Gap: {gap:.4f}")

    wandb.log({
        'epoch': epoch + 1,
        'train/total_loss': avg_loss,
        'train/win_loss': avg_win_loss,
        'train/injury_loss': avg_injury_loss,
        'val/total_loss': avg_val_loss,
        'val/win_loss': avg_val_win_loss,
        'val/injury_loss': avg_val_injury_loss,
        'learning_rate': current_lr,
        'train_val_gap': gap,
        'epochs_no_improve': epochs_no_improve
    })


    print("\nDetailed Validation Metrics:")
    val_metrics, inj_t, inj_p = evaluate_metrics(model, val_loader, device)

    print(f"Win - Acc: {val_metrics['win_accuracy']:.4f} | AUC: {val_metrics['win_auc']:.4f}")
    print(f"Injury - Acc: {val_metrics['injury_accuracy']:.4f} | AUC: {val_metrics['injury_auc']:.4f}")
    print(f"Injury - Precision: {val_metrics['injury_precision']:.4f} | Recall: {val_metrics['injury_recall']:.4f} | F1: {val_metrics['injury_f1']:.4f}")

    wandb.log({
        "val/win_accuracy": val_metrics["win_accuracy"],
        "val/win_auc": val_metrics["win_auc"],
        "val/injury_accuracy": val_metrics["injury_accuracy"],
        "val/injury_auc": val_metrics["injury_auc"],
        "val/injury_precision": val_metrics["injury_precision"],
        "val/injury_recall": val_metrics["injury_recall"],
        "val/injury_f1": val_metrics["injury_f1"],
    })

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_loss,
        'val_loss': avg_val_loss,
        'train_win_loss': avg_win_loss,
        'train_injury_loss': avg_injury_loss,
        'val_win_loss': avg_val_win_loss,
        'val_injury_loss': avg_val_injury_loss,
    }
    torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pth")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(checkpoint, checkpoint_dir / "best_model.pth")
        print(f"New best model saved. Val loss: {avg_val_loss:.4f}")

        wandb.run.summary['best_val_loss'] = best_val_loss
        wandb.run.summary['best_epoch'] = best_epoch

    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs")

    if (epoch + 1) % 5 == 0:
        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth")

    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

    if epochs_no_improve >= patience:
        wandb.run.summary['stopped_early'] = True
        wandb.run.summary['total_epochs_trained'] = epoch + 1
        break
    else:
        wandb.run.summary['stopped_early'] = False
        wandb.run.summary['total_epochs_trained'] = n_epochs

    print(f"Best validation loss: {best_val_loss:.4f}")


best_checkpoint = torch.load(checkpoint_dir / "best_model.pth")
model.load_state_dict(best_checkpoint['model_state_dict'])
model.to(device)
model.eval()

test_metrics, inj_t, inj_p = evaluate_metrics(model, test_loader, device)
print("\nTest Set Results:")
print(f"Win Prediction:")
print(f"Accuracy: {test_metrics['win_accuracy']:.4f}")
print(f"AUC-ROC: {test_metrics['win_auc']:.4f}")

print(f"\nInjury Prediction:")
print(f"Accuracy: {test_metrics['injury_accuracy']:.4f}")
print(f"AUC-ROC: {test_metrics['injury_auc']:.4f}")
print(f"Precision: {test_metrics['injury_precision']:.4f}")
print(f"Recall: {test_metrics['injury_recall']:.4f}")
print(f"F1-Score: {test_metrics['injury_f1']:.4f}")

risk_results, _, _ = evaluate_risk_tiers(model, test_loader, device)
optimal_thresh, best_prec, best_rec = find_threshold_at_recall(
    inj_t, inj_p, min_recall=0.4
)
print(f"Optimal threshold: {optimal_thresh:.3f} (Precision: {best_prec:.1%}, Recall: {best_rec:.1%})")
test_metrics_opt, _, _ = evaluate_metrics(model, test_loader, device, injury_threshold=optimal_thresh)
print(f"\nWith optimal threshold ({optimal_thresh:.2f}):")
print(f"  Precision: {test_metrics_opt['injury_precision']:.1%}")
print(f"  Recall: {test_metrics_opt['injury_recall']:.1%}")
print(f"  F1: {test_metrics_opt['injury_f1']:.3f}")


wandb.log({
    "test/win_accuracy": test_metrics["win_accuracy"],
    "test/win_auc": test_metrics["win_auc"],
    "test/injury_accuracy": test_metrics["injury_accuracy"],
    "test/injury_auc": test_metrics["injury_auc"],
    "test/injury_precision": test_metrics["injury_precision"],
    "test/injury_recall": test_metrics["injury_recall"],
    "test/injury_f1": test_metrics["injury_f1"],
})

wandb.run.summary.update({
    "final_test_win_acc": test_metrics["win_accuracy"],
    "final_test_win_auc": test_metrics["win_auc"],
    "final_test_injury_auc": test_metrics["injury_auc"],
    "final_test_injury_f1": test_metrics["injury_f1"],
})

artifact = wandb.Artifact('best-model', type='model')
artifact.add_file(str(checkpoint_dir / 'best_model.pth'))
wandb.log_artifact(artifact)

wandb.finish()

val_metrics, inj_targets, inj_probs = evaluate_metrics(model, val_loader, device)

optimal_thresh, best_f1 = find_optimal_threshold(inj_targets, inj_probs)
print(f"Optimal threshold: {optimal_thresh:.2f} (F1: {best_f1:.4f})")

test_metrics, _, _ = evaluate_metrics(model, test_loader, device, injury_threshold=optimal_thresh)

print(f"\nWin Prediction:")
print(f"  Accuracy: {test_metrics['win_accuracy']:.4f}")
print(f"  AUC-ROC:  {test_metrics['win_auc']:.4f}")

print(f"\nInjury Prediction (threshold={optimal_thresh:.2f}):")
print(f"Accuracy: {test_metrics['injury_accuracy']:.4f}")
print(f"AUC-ROC: {test_metrics['injury_auc']:.4f}")
print(f"Avg Prec: {test_metrics['injury_avg_precision']:.4f}")
print(f"Precision: {test_metrics['injury_precision']:.4f}")
print(f"Recall: {test_metrics['injury_recall']:.4f}")
print(f"F1-Score: {test_metrics['injury_f1']:.4f}")
print(f"\nInjury Rate (actual): {test_metrics['injury_rate']:.2%}")
print(f"Injury Rate (pred): {test_metrics['injury_pred_rate']:.2%}")
print(f"TP: {test_metrics['injury_tp']}, FP: {test_metrics['injury_fp']}, "
      f"TN: {test_metrics['injury_tn']}, FN: {test_metrics['injury_fn']}")