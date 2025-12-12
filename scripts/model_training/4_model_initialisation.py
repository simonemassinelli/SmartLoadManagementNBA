import torch
from src.model.SmartLoadModel import SmartLoadModel
from src.data.GameDataset import GameDataset

def count_params(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SmartLoadModel(
    n_shared_features=
)