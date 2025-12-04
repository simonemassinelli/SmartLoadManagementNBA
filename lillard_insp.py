import pandas as pd

merged = pd.read_csv(r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\players_games_w_ratings.csv")
merged['GAME_DATE_EST'] = pd.to_datetime(merged['GAME_DATE_EST'])
lillard1 = merged.loc[(merged['PLAYER_NAME']=='Damian Lillard') & (merged['GAME_DATE_EST'].dt.year == 2020)]
print(lillard1)

lillard2 = merged.loc[(merged['PLAYER_NAME']=='Damian Lillard') & (merged['SEASON'] == 20)]
print(lillard2)

print(len(lillard1), len(lillard2))