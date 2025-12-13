import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


def evaluate_model(model, dataloader, device):
    model.eval()
    win_preds, win_labels = [], []
    inj_preds, inj_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            preds = model(batch)

            win_prob = torch.sigmoid(preds["win_logits"]).squeeze(-1)
            win_preds.extend(win_prob.cpu().numpy())
            win_labels.extend(batch["won"].cpu().numpy())

            inj_prob = torch.sigmoid(preds["injury_logits"])
            inj_mask = batch["player_mask"].bool()

            inj_preds.extend(inj_prob[inj_mask].cpu().numpy())
            inj_labels.extend(batch["injuries"][inj_mask].cpu().numpy())

    win_acc = accuracy_score(win_labels, (np.array(win_preds) >= 0.5).astype(int))
    win_auc = roc_auc_score(win_labels, win_preds)

    inj_acc = accuracy_score(inj_labels, (np.array(inj_preds) >= 0.5).astype(int))
    inj_auc = roc_auc_score(inj_labels, inj_preds)

    return {
        "win_acc": win_acc,
        "win_auc": win_auc,
        "inj_acc": inj_acc,
        "inj_auc": inj_auc,
    }


def injury_classification_metrics(model, dataloader, device):
    model.eval()
    probs_all, targets_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            preds = model(batch)
            probs = torch.sigmoid(preds["injury_logits"])
            mask = batch["player_mask"].bool()

            probs_all.extend(probs[mask].cpu().numpy())
            targets_all.extend(batch["injuries"][mask].cpu().numpy())

    preds_bin = (np.array(probs_all) > 0.5).astype(int)

    precision = precision_score(targets_all, preds_bin)
    recall = recall_score(targets_all, preds_bin)
    f1 = f1_score(targets_all, preds_bin)

    return precision, recall, f1