import pandas as pd
import unicodedata

path_main = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\simone_cleaned.csv"
path_seasons = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\all_seasons.csv"
path_players = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\players.csv"
path_roles_hist = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\NBA Player Stats(1950 - 2022).csv"


df_main = pd.read_csv(path_main)
df_seasons = pd.read_csv(path_seasons)
df_pos_info = pd.read_csv(path_players, encoding='latin1', delimiter=';')
df_roles_hist = pd.read_csv(path_roles_hist)

#Cleaning the names to merge the datasets correctly

def clean_name(name):
    if not isinstance(name, str):
        return str(name)


    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    name = name.replace('*', '').replace('.', '')
    name = name.strip()
    return name.lower()


df_main['PLAYER_NAME_CLEANED'] = df_main['PLAYER_NAME'].apply(clean_name)
df_seasons['player_merge_key'] = df_seasons['player_name'].apply(clean_name)
df_pos_info['player_merge_key'] = df_pos_info['Player'].apply(clean_name)

df_roles_unique = df_pos_info.sort_values('Year').drop_duplicates(subset=['player_merge_key'], keep='last')
df_roles_unique = df_roles_unique[['player_merge_key', 'Pos', 'GS']]

df_features = pd.merge(df_seasons, df_roles_unique, on='player_merge_key', how='left')

df_features = df_features[(df_features['season'] >= '2008-09') & (df_features['season'] <= '2018-19')].copy()


season_map_1 = {
    '2008-09': 2009, '2009-10': 2010, '2010-11': 2011, '2011-12': 2012,
    '2012-13': 2013, '2013-14': 2014, '2014-15': 2015, '2015-16': 2016,
    '2016-17': 2017, '2017-18': 2018, '2018-19': 2019
}
df_features['season_year'] = df_features['season'].map(season_map_1)


season_map_2 = {
    2009: 9, 2010: 10, 2011: 11, 2012: 12, 2013: 13,
    2014: 14, 2015: 15, 2016: 16, 2017: 17, 2018: 18, 2019: 19
}
df_features['SEASON_MATCH'] = df_features['season_year'].map(season_map_2)

print(f"Features pronte. Shape: {df_features.shape}")

#Merging datasets using the cleaned names and 2-digits seasons as keys
df_merged = pd.merge(
    df_main,
    df_features,
    left_on=['PLAYER_NAME_CLEANED', 'SEASON'],
    right_on=['player_merge_key', 'SEASON_MATCH'],
    how='left'
)


cols_to_drop = [
    'Unnamed: 0', 'player_merge_key', 'season', 'season_year', 'SEASON_MATCH',
    'team_abbreviation', 'college', 'country', 'draft_round', 'draft_number',
    'pts', 'reb', 'ast', 'ts_pct', 'ast_pct'
]

cols_to_drop = [c for c in cols_to_drop if c in df_merged.columns]
df_merged.drop(columns=cols_to_drop, inplace=True)

print(f"Merge completato. Shape: {df_merged.shape}")

#cleaning df_roles before the merge

df_roles_hist = df_roles_hist[(df_roles_hist['Season']>=2009) & (df_roles_hist['Season']<=2019)].copy()
df_roles_hist['Player_CLEANED'] = df_roles_hist['Player'].apply(clean_name)
df_roles_hist['Season_2_DIGITS'] = df_roles_hist['Season'].astype(str).str[-2:].astype(int)


df_roles_hist_unique = df_roles_hist.drop_duplicates(subset=['Player_CLEANED', 'Season_2_DIGITS'], keep='first')

df_merged_final = pd.merge(
    df_merged,
    df_roles_hist_unique[['Player_CLEANED', 'Season_2_DIGITS', 'Pos']],
    left_on=['PLAYER_NAME_CLEANED', 'SEASON'],
    right_on=['Player_CLEANED', 'Season_2_DIGITS'],
    how='left',
    suffixes=('', '_hist')
)


if 'Pos' in df_merged_final.columns and 'Pos_hist' in df_merged_final.columns:
    df_merged_final['Pos'] = df_merged_final['Pos'].fillna(df_merged_final['Pos_hist'])


df_merged_final.drop(columns=['Player_CLEANED', 'Season_2_DIGITS', 'Pos_hist'], inplace=True, errors='ignore')

#Renaming variables
rename_dict = {
    'age': 'AGE', 'player_height': 'PLAYER_HEIGHT', 'player_weight': 'PLAYER_WEIGHT',
    'gp': 'GP', 'net_rating': 'NET_RATING', 'usg_pct': 'USG_PCT', 'Pos': 'POS',
    'oreb_pct': 'OREB_PCT', 'dreb_pct': 'DREB_PCT', 'draft_year': 'DRAFT_YEAR'
}
df_merged_final.rename(columns=rename_dict, inplace=True)

df = df_merged_final.copy()

#Player rating of the season
player_seasonal_rating = df[['SEASON', 'PLAYER_NAME', 'PLAYER_TEAM', 'RATING']].drop_duplicates()
player_seasonal_rating = player_seasonal_rating.groupby(['SEASON', 'PLAYER_NAME', 'PLAYER_TEAM'])['RATING'].mean().reset_index() # Uso mean per sicurezza su duplicati
player_seasonal_rating.rename(columns={'RATING': 'player_seasonal_rating'}, inplace=True)

#Sum of the ratings of the players in the same team  for each season
team_season_total = player_seasonal_rating.groupby(['SEASON', 'PLAYER_TEAM'])['player_seasonal_rating'].sum().reset_index()
team_season_total.rename(columns={'player_seasonal_rating': 'team_season_total_rating'}, inplace=True)

#Merging the info
player_data = pd.merge(player_seasonal_rating, team_season_total, on=['SEASON', 'PLAYER_TEAM'], how='left')

#Computing Ratio
player_data['PLAYER_IMPORTANCE'] = ((player_data['player_seasonal_rating'] / player_data['team_season_total_rating']) * 100).round(2)

#Merge back into the original DF
df = pd.merge(df, player_data[['SEASON', 'PLAYER_NAME', 'PLAYER_TEAM', 'PLAYER_IMPORTANCE']],
              on=['SEASON', 'PLAYER_NAME', 'PLAYER_TEAM'], how='left')

# CORREZIONE: errors='ignore' per evitare crash se una colonna non esiste
cols_to_clean = ['Unnamed: 0.1', 'Unnamed: 0_x', 'PLAYER_NAME_clean', 'PLAYER_NAME_CLEANED', 'Unnamed: 0_y', 'player_name']
df.drop(columns=cols_to_clean, inplace=True, errors='ignore')

print(len(df))
df.to_csv('DS_with_Ratings.csv', index=False)
print("File saved")