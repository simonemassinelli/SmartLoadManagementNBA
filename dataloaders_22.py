import pandas as pd

from updates.GameDataset import GameDataset
from torch.utils.data import DataLoader, random_split, Subset


# Time based split
def get_dataloaders(csv_path, batch_size, train_split, val_split):
    dataset = GameDataset(csv_path)

    dates = sorted(set(
        pd.Timestamp(game['date']).normalize()
        for game in dataset.games
    ))

    print(f"Total games: {len(dataset.games):,}")
    print(f"Unique dates: {len(dates):,}")
    print(f"Avg games/day: {len(dataset.games) / len(dates):.1f}")

    n_dates = len(dates)
    n_train_dates = int(train_split * n_dates)
    n_val_dates = int(val_split * n_dates)

    train_dates = set(dates[:n_train_dates])
    val_dates = set(dates[n_train_dates:n_train_dates + n_val_dates])
    test_dates = set(dates[n_train_dates + n_val_dates:])

    train_idx = []
    val_idx = []
    test_idx = []

    for i, game in enumerate(dataset.games):
        game_date = pd.Timestamp(game['date']).normalize()

        if game_date in train_dates:
            train_idx.append(i)
        elif game_date in val_dates:
            val_idx.append(i)
        else:
            test_idx.append(i)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True,  num_workers=0, pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory = True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds):,} games from {len(train_dates):,} dates")
    print(f"({dates[0].date()} - {dates[n_train_dates - 1].date()})")
    print(f"Val: {len(val_ds):,} games from {len(val_dates):,} dates")
    print(f"({dates[n_train_dates].date()} - {dates[n_train_dates + n_val_dates - 1].date()})")
    print(f"Test: {len(test_ds):,} games from {len(test_dates):,} dates")
    print(f"({dates[n_train_dates + n_val_dates].date()} - {dates[-1].date()})")

    return train_loader, val_loader, test_loader