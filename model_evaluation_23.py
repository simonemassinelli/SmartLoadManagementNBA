import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)


def evaluate_metrics(model, dataloader, device, injury_threshold=0.5):
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
    injury_preds = (injury_probs >= injury_threshold).astype(int)

    metrics = {
        "win_accuracy": accuracy_score(win_targets, win_preds),
        "injury_accuracy": accuracy_score(injury_targets, injury_preds),
        "injury_precision": precision_score(injury_targets, injury_preds, zero_division=0),
        "injury_recall": recall_score(injury_targets, injury_preds, zero_division=0),
        "injury_f1": f1_score(injury_targets, injury_preds, zero_division=0),
    }

    try:
        metrics["win_auc"] = roc_auc_score(win_targets, win_probs)
    except ValueError:
        metrics["win_auc"] = 0.5

    try:
        metrics["injury_auc"] = roc_auc_score(injury_targets, injury_probs)
        metrics["injury_avg_precision"] = average_precision_score(injury_targets, injury_probs)
    except ValueError:
        metrics["injury_auc"] = 0.5
        metrics["injury_avg_precision"] = 0.0

    tn, fp, fn, tp = confusion_matrix(injury_targets, injury_preds, labels=[0, 1]).ravel()
    metrics["injury_tp"] = int(tp)
    metrics["injury_fp"] = int(fp)
    metrics["injury_tn"] = int(tn)
    metrics["injury_fn"] = int(fn)

    metrics["injury_rate"] = injury_targets.mean()
    metrics["injury_pred_rate"] = injury_preds.mean()

    return metrics, injury_targets, injury_probs


def find_optimal_threshold(injury_targets, injury_probs, metric='f1'):
    thresholds = np.arange(0.05, 0.5, 0.01)
    best_threshold = 0.5
    best_score = 0

    for thresh in thresholds:
        preds = (injury_probs >= thresh).astype(int)

        if metric == 'f1':
            score = f1_score(injury_targets, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(injury_targets, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def find_threshold_at_recall(injury_targets, injury_probs, min_recall=0.5):
    precision, recall, thresholds = precision_recall_curve(injury_targets, injury_probs)

    valid_idx = recall[:-1] >= min_recall

    if valid_idx.sum() == 0:
        print(f"Warning: Cannot achieve {min_recall:.0%} recall. Max recall: {recall.max():.2%}")
        return 0.1, 0, recall.max()

    best_idx = np.where(valid_idx)[0][np.argmax(precision[:-1][valid_idx])]

    best_thresh = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    return best_thresh, best_precision, best_recall


def evaluate_risk_tiers(model, dataloader, device):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            preds = model(batch)
            mask = batch["player_mask"].bool()

            probs = torch.sigmoid(preds["injury_logits"])[mask]
            targets = batch["injuries"].float()[mask]

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    high_risk = probs >= 0.15
    medium_risk = (probs >= 0.05) & (probs < 0.15)
    low_risk = probs < 0.05

    base_rate = targets.mean()

    print(f"Base injury rate: {base_rate:.2%})")

    results = {}
    for name, tier_mask in [("High", high_risk), ("Medium", medium_risk), ("Low", low_risk)]:
        count = tier_mask.sum()
        pct = count / len(targets) * 100
        rate = targets[tier_mask].mean() if count > 0 else 0
        lift = rate / base_rate if base_rate > 0 else 0

        print(f"{name:<12} {count:<10} {pct:<10.1f} {rate:<12.2%} {lift:<8.1f}x")

        results[name.lower()] = {
            "count": int(count),
            "pct": pct,
            "injury_rate": rate,
            "lift": lift
        }

    return results, probs, targets