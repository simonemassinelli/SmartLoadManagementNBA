import pandas as pd
import os

dataset_path = r"C:\Users\casam\.cache\kagglehub\datasets\nathanlauga\nba-games\versions\10"

games_details_path = os.path.join(dataset_path, "games_details.csv")
games_details_df = pd.read_csv(games_details_path)

games_path = os.path.join(dataset_path, "games.csv")
games_df = pd.read_csv(games_path)

merged = games_df.merge(games_details_df, on='GAME_ID', how='left')

simpson = merged[merged['PLAYER_NAME']=='Zavier Simpson']
print(simpson['GAME_DATE_EST'])
print(simpson.iloc[3])
print(simpson.iloc[4])
print(simpson.iloc[5])
print(simpson.iloc[6])