import pandas as pd
import os
import numpy as np

dataset_path = r"C:\Users\casam\.cache\kagglehub\datasets\nathanlauga\nba-games\versions\10"

rating_csv = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\all_players_ratings_normalized.csv"
ratings_df = pd.read_csv(rating_csv, dtype={'Name': str})

games_path = os.path.join(dataset_path, "games.csv")
games_df = pd.read_csv(games_path,
    dtype={'GAME_DATE_EST': str,  'GAME_ID': str, 'HOME_TEAM_ID': int, 'VISITOR_TEAM_ID': int,
        'PTS_home': float, 'PTS_away': float, 'HOME_TEAM_WINS': int})
games_df_necessary = games_df[['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                               'PTS_home', 'PTS_away', 'HOME_TEAM_WINS']]

games_details_path = os.path.join(dataset_path, "games_details.csv")
games_details_df = pd.read_csv(games_details_path,
    dtype={'GAME_ID': str, 'TEAM_ID': int, 'PLAYER_NAME': str, 'COMMENT': str, 'MIN': str, 'PLUS_MINUS': int})
games_details_df_necessary = games_details_df[['GAME_ID', 'TEAM_ID', 'PLAYER_NAME',
                                               'COMMENT', 'MIN', 'PLUS_MINUS']]

teams_path = os.path.join(dataset_path, "teams.csv")
teams_df = pd.read_csv(teams_path,
    dtype={'TEAM_ID': int, 'ABBREVIATION': str})
teams_df_necessary = teams_df[['TEAM_ID', 'ABBREVIATION']]

merged = games_df_necessary.merge(games_details_df_necessary, on='GAME_ID', how='left')

merged = merged.merge(
    teams_df_necessary.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "ABBREVIATION": "HOME_TEAM"}),
    on="HOME_TEAM_ID",
    how="left")

merged = merged.merge(
    teams_df_necessary.rename(columns={"TEAM_ID": "VISITOR_TEAM_ID", "ABBREVIATION": "AWAY_TEAM"}),
    on="VISITOR_TEAM_ID",
    how="left")

merged = merged.merge(
    teams_df_necessary.rename(columns={"TEAM_ID": "TEAM_ID", "ABBREVIATION": "PLAYER_TEAM"}),
    on="TEAM_ID",
    how="left")

merged = merged.drop(columns=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'TEAM_ID'])

merged['MIN'] = merged['MIN'].fillna('00:00')
merged = merged.dropna(subset=['PLAYER_TEAM'])

merged['POINTS_DIFF'] = np.where(
    merged['PLAYER_TEAM'] == merged['HOME_TEAM'],
    merged['PTS_home'] - merged['PTS_away'],
    merged['PTS_away'] - merged['PTS_home'])

no_colon = merged[~merged['MIN'].str.contains(':')].copy()
no_colon['MIN'] = no_colon['MIN'] + ':00'
merged.update(no_colon)

def minutes_to_int(min_str):
    mins, secs = str(min_str).split(':')
    mins = int(float(mins))
    secs = int(secs)
    if secs >= 30:
        mins += 1
    return mins

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

ratings_df['Name_clean'] = ratings_df['Name'].str.lower().str.replace('.', '').str.replace(' ', '')
merged['PLAYER_NAME_clean'] = merged['PLAYER_NAME'].str.lower().str.replace('.', '').str.replace(' ', '')

rating_year_cols = [col for col in ratings_df.columns if '_norm' in col]

def get_rating(row):
    season = str(row['SEASON'])
    look_for = season + '_norm'
    if look_for in rating_year_cols:
        matched = ratings_df.loc[ratings_df['Name_clean'] == row['PLAYER_NAME_clean'], look_for]
        if not matched.empty:
            return matched.values[0]
    return np.nan

merged['RATING'] = merged.apply(get_rating, axis=1)

def fill_previous_rating(row):
    if pd.notna(row['RATING']):
        return row['RATING']
    season = row['SEASON'] - 1
    look_for = str(season) + '_norm'
    if look_for in rating_year_cols:
        matched = ratings_df.loc[ratings_df['Name_clean'] == row['PLAYER_NAME_clean'], look_for]
        if not matched.empty:
            return matched.values[0]
    return np.nan

def fill_next_rating(row):
    if pd.notna(row['RATING']):
        return row['RATING']
    season = row['SEASON'] + 1
    look_for = str(season) + '_norm'
    if look_for in rating_year_cols:
        matched = ratings_df.loc[ratings_df['Name_clean'] == row['PLAYER_NAME_clean'], look_for]
        if not matched.empty:
            return matched.values[0]
    return np.nan

merged['RATING'] = merged.apply(fill_previous_rating, axis=1)
merged['RATING'] = merged.apply(fill_next_rating, axis=1)

merged.to_csv("players_games_w_ratings.csv", index=False)