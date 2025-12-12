from src.data.GameDataset import GameDataset
from torch.utils.data import DataLoader, random_split


def get_dataloaders(csv_path, batch_size, train_split, val_split):
    dataset = GameDataset(csv_path)

    print(len(dataset))


path = '../../data/nba_game_features_final.csv'
get_dataloaders(path, 32, 0.8, 0.1)