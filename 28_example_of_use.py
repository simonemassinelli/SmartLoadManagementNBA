import pandas as pd
import numpy as np
import torch
from SmartLoadModel_19 import SmartLoadModel
from features_20 import SHARED_FEATURES, PLAYER_FEATURES, WIN_FEATURES, INJURY_FEATURES
from objf_24 import optimize_minutes_today

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica CSV
path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\nba_game_features_final.csv"
df = pd.read_csv(path)
df_sorted = df.sort_values('GAME_ID')

# Seleziona una partita
game_id = df_sorted['GAME_ID'].iloc[50000]
players_in_game = df_sorted[df_sorted['GAME_ID'] == game_id]
players_in_game = players_in_game.sort_values('RATING')

max_players = 19
num_players = len(players_in_game)

def pad_features(df, features, max_players):
    arr = df[features].values.astype(float)
    padded = np.zeros((max_players, arr.shape[1]))
    padded[:arr.shape[0], :] = arr
    return padded

# Preparazione input modello
game_input = {
    'player_features': torch.tensor(pad_features(players_in_game, PLAYER_FEATURES, max_players), dtype=torch.float32).unsqueeze(0).to(device),
    'shared_features': torch.tensor(players_in_game[SHARED_FEATURES].iloc[0].values.astype(float), dtype=torch.float32).unsqueeze(0).to(device),
    'win_features': torch.tensor(players_in_game[WIN_FEATURES].iloc[0].values.astype(float), dtype=torch.float32).unsqueeze(0).to(device),
    'injury_features': torch.tensor(pad_features(players_in_game, INJURY_FEATURES, max_players), dtype=torch.float32).unsqueeze(0).to(device),
    'player_mask': torch.zeros((1, max_players), dtype=torch.float32).to(device),
    'actual_minutes': torch.zeros((1, max_players), dtype=torch.float32).to(device),
    'MIN_INT': torch.zeros((1, max_players), dtype=torch.float32).to(device)
}

# Set mask e minuti reali solo per i giocatori effettivi
game_input['player_mask'][0, :num_players] = 1
game_input['actual_minutes'][0, :num_players] = torch.tensor(players_in_game['MIN_INT'].values.astype(float))
game_input['MIN_INT'][0, :num_players] = torch.tensor(players_in_game['MIN_INT'].values.astype(float))

# Inizializza modello
model = SmartLoadModel(
    n_shared_features=len(SHARED_FEATURES),
    n_player_features=len(PLAYER_FEATURES),
    n_win_features=len(WIN_FEATURES),
    n_injury_features=len(INJURY_FEATURES),
    hidden_dim=256,
    n_attention_heads=4,
    dropout=0.3
).to(device)

# Carica checkpoint
checkpoint_path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\smartload_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Funzione per generare combinazioni di minuti
import itertools

def generate_candidate_minutes(n_players, min_min=20, max_min=40, step=4, n_samples=20):
    possible_values = list(range(min_min, max_min + 1, step))
    candidates = []
    for _ in range(n_samples):
        sample = np.random.choice(possible_values, size=n_players, replace=True)
        candidates.append(sample)
    return candidates

# Candidate allocation minuti per i giocatori della partita
candidate_minutes_list = generate_candidate_minutes(num_players, n_samples=50)

# Esegui ottimizzazione
best_minutes, best_score = optimize_minutes_today(
    model=model,
    game_input=game_input,
    candidate_minutes_list=candidate_minutes_list,
    n_simulations=500,
    device=device
)

print("Miglior allocazione minuti:", best_minutes)
print("Obiettivo massimo:", best_score)