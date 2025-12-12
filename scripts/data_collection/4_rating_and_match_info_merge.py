import pandas as pd
import os
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

dataset_path = r"C:\Users\casam\.cache\kagglehub\datasets\nathanlauga\nba-games\versions\10"
rating_csv = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\all_players_ratings_normalized.csv"

ratings_df = pd.read_csv(rating_csv, dtype={'Name': str})

games_path = os.path.join(dataset_path, "games.csv")
games_df = pd.read_csv(games_path, low_memory=False,
    dtype={'GAME_ID': str, 'HOME_TEAM_ID': int, 'VISITOR_TEAM_ID': int,
           'PTS_home': float, 'PTS_away': float})

games_details_path = os.path.join(dataset_path, "games_details.csv")
games_details_df = pd.read_csv(games_details_path, low_memory=False,
    dtype={'GAME_ID': str, 'TEAM_ID': int, 'PLAYER_NAME': str, 'COMMENT': str, 'MIN': str, 'START_POSITION': str})

teams_path = os.path.join(dataset_path, "teams.csv")
teams_df = pd.read_csv(teams_path, dtype={'TEAM_ID': int, 'ABBREVIATION': str})

games_df_necessary = games_df[['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                               'PTS_home', 'PTS_away']]
games_details_df_necessary = games_details_df[['GAME_ID', 'TEAM_ID', 'PLAYER_NAME',
                                               'COMMENT', 'MIN', 'PLUS_MINUS', 'START_POSITION']]
teams_df_necessary = teams_df[['TEAM_ID', 'ABBREVIATION']]

merged = games_df_necessary.merge(games_details_df_necessary, on='GAME_ID', how='left')

merged = merged.merge(
    teams_df_necessary.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "ABBREVIATION": "HOME_TEAM"}),
    on="HOME_TEAM_ID", how="left")

merged = merged.merge(
    teams_df_necessary.rename(columns={"TEAM_ID": "VISITOR_TEAM_ID", "ABBREVIATION": "AWAY_TEAM"}),
    on="VISITOR_TEAM_ID", how="left")

merged = merged.merge(
    teams_df_necessary.rename(columns={"TEAM_ID": "TEAM_ID", "ABBREVIATION": "PLAYER_TEAM"}),
    on="TEAM_ID", how="left")

merged = merged.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'TEAM_ID'])

merged['MIN'] = merged['MIN'].fillna('00:00')
merged = merged.dropna(subset=['PLAYER_TEAM'])

merged['PLUS_MINUS'] = pd.to_numeric(merged['PLUS_MINUS'], errors='coerce').fillna(0).astype(int)

merged['POINTS_DIFF'] = np.where(
    merged['PLAYER_TEAM'] == merged['HOME_TEAM'],
    merged['PTS_home'] - merged['PTS_away'],
    merged['PTS_away'] - merged['PTS_home'])

mask_no_colon = ~merged['MIN'].str.contains(':')
merged.loc[mask_no_colon, 'MIN'] = merged.loc[mask_no_colon, 'MIN'] + ':00'

def minutes_to_int(min_str):
    try:
        parts = str(min_str).split(':')
        if len(parts) == 2:
            mins = int(float(parts[0]))
            secs = int(float(parts[1]))
            if secs >= 30:
                mins += 1
            return mins
        return 0
    except:
        return 0

merged['MIN_INT'] = merged['MIN'].apply(minutes_to_int)
merged = merged.drop(columns=['MIN'])

merged['GAME_DATE_EST'] = pd.to_datetime(merged['GAME_DATE_EST'], errors='coerce')

year = merged['GAME_DATE_EST'].dt.year
month = merged['GAME_DATE_EST'].dt.month
merged['SEASON_FULL'] = year + (month >= 9).astype(int)
merged['SEASON'] = merged['SEASON_FULL'] % 100
merged = merged.drop('SEASON_FULL', axis=1)

merged = merged[(merged['SEASON'] < 20) & (merged['SEASON'] > 5)].copy()
merged = merged[~((merged['GAME_DATE_EST'].dt.month >= 7) & (merged['GAME_DATE_EST'].dt.month <= 9)) &
                ~((merged['GAME_DATE_EST'].dt.month == 10) & (merged['GAME_DATE_EST'].dt.day < 15))]

merged['IS_STARTER'] = np.where(merged['START_POSITION'].notna(), 1, 0)
merged['PLAYED'] = np.where(merged['COMMENT'].isna(), 1, 0)
merged['GS_real'] = merged.groupby(['PLAYER_NAME', 'SEASON'])['IS_STARTER'].transform('sum')
merged['GP_real'] = merged.groupby(['PLAYER_NAME', 'SEASON'])['PLAYED'].transform('sum')
merged = merged.drop(columns=['START_POSITION', 'IS_STARTER'])

ratings_df['Name_clean'] = ratings_df['Name'].str.lower().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)
merged['PLAYER_NAME_clean'] = merged['PLAYER_NAME'].str.lower().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)

rating_cols = [col for col in ratings_df.columns if '_norm' in col]

ratings_melted = ratings_df.melt(id_vars=['Name_clean'], value_vars=rating_cols, var_name='SEASON_STR', value_name='RATING_VAL')
ratings_melted['SEASON_INT'] = ratings_melted['SEASON_STR'].str.replace('_norm', '', regex=False).astype(int)

ratings_lookup = ratings_melted[['Name_clean', 'SEASON_INT', 'RATING_VAL']].dropna()

merged = merged.merge(
    ratings_lookup,
    left_on=['PLAYER_NAME_clean', 'SEASON'],
    right_on=['Name_clean', 'SEASON_INT'],
    how='left'
)
merged = merged.rename(columns={'RATING_VAL': 'RATING'})
merged = merged.drop(columns=['Name_clean', 'SEASON_INT'])

ratings_prev = ratings_lookup.copy()
ratings_prev['SEASON_MATCH'] = ratings_prev['SEASON_INT'] + 1
ratings_prev = ratings_prev.rename(columns={'RATING_VAL': 'RATING_PREV'})

ratings_next = ratings_lookup.copy()
ratings_next['SEASON_MATCH'] = ratings_next['SEASON_INT'] - 1
ratings_next = ratings_next.rename(columns={'RATING_VAL': 'RATING_NEXT'})

merged = merged.merge(
    ratings_prev[['Name_clean', 'SEASON_MATCH', 'RATING_PREV']],
    left_on=['PLAYER_NAME_clean', 'SEASON'],
    right_on=['Name_clean', 'SEASON_MATCH'],
    how='left'
)

merged = merged.merge(
    ratings_next[['Name_clean', 'SEASON_MATCH', 'RATING_NEXT']],
    left_on=['PLAYER_NAME_clean', 'SEASON'],
    right_on=['Name_clean', 'SEASON_MATCH'],
    how='left'
)

merged['RATING'] = merged['RATING'].fillna(merged['RATING_PREV']).fillna(merged['RATING_NEXT'])

cols_to_drop = ['Name_clean_x', 'Name_clean_y', 'SEASON_MATCH_x', 'SEASON_MATCH_y',
                'RATING_PREV', 'RATING_NEXT', 'PLAYER_NAME_clean', 'GP']
cols_to_drop = [c for c in cols_to_drop if c in merged.columns]
merged = merged.drop(columns=cols_to_drop)

print(len(merged))
merged.to_csv("players_games_w_ratings.csv", index=False)