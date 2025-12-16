import pandas as pd
import numpy as np
import torch
from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
from objf_27 import optimize_minutes_evolutionary, AVG_INJURY_GAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_features(df, features, max_players):
    """Pad player features to max_players slots with zeros."""
    arr = df[features].values.astype(float)
    padded = np.zeros((max_players, arr.shape[1]))
    padded[:arr.shape[0], :] = arr
    return padded

# Load Data
path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\nba_game_features_final.csv"
df = pd.read_csv(path)

# Filter Game and Players
game_id = 21700231
players_in_game = df[df['GAME_ID'] == game_id]
gsw_players = players_in_game[(players_in_game['PLAYER_TEAM'] == 'GSW') & (players_in_game['IS_INJURED'] == 0)]
gsw_players = gsw_players.sort_values('RATING', ascending=False)
player_names = gsw_players['PLAYER_NAME'].values

num_players = len(gsw_players)
max_players = max(num_players, 19)

# Prepare Tensors
game_input = {
    'player_features': torch.tensor(pad_features(gsw_players, PLAYER_FEATURES, max_players), dtype=torch.float32).unsqueeze(0).to(device),
    'shared_features': torch.tensor(gsw_players[SHARED_FEATURES].iloc[0].values.astype(float), dtype=torch.float32).unsqueeze(0).to(device),
    'win_features': torch.tensor(gsw_players[WIN_FEATURES].iloc[0].values.astype(float), dtype=torch.float32).unsqueeze(0).to(device),
    'injury_features': torch.tensor(pad_features(gsw_players, INJURY_FEATURES, max_players), dtype=torch.float32).unsqueeze(0).to(device),
    'player_mask': torch.zeros((1, max_players), dtype=torch.float32).to(device),
    'actual_minutes': torch.zeros((1, max_players), dtype=torch.float32).to(device),
    'MIN_INT': torch.zeros((1, max_players), dtype=torch.float32).to(device)
}
game_input['player_mask'][0, :num_players] = 1

# Load Model
model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
    hidden_dim=256,
    n_attention_heads=4,
    dropout=0.3
).to(device)

checkpoint_path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run Optimization
best_minutes, best_score = optimize_minutes_evolutionary(
    model=model,
    game_input=game_input,
    num_players=num_players,
    max_players=max_players,
    generations=20,
    population_size=50,
    n_sims_per_eval=50,
    device=device
)

# Output Results
print("\nOPTIMIZATION COMPLETE")
print("Best minute allocation (GSW active players):")
for i in range(num_players):
    if best_minutes[i] > 0:
        print(f"{player_names[i]}: {int(best_minutes[i])} min")

print(f"\nMaximum objective value: {best_score:.4f}")
print(f"(Includes win prob today + sum of expected wins over next {AVG_INJURY_GAMES} games)")