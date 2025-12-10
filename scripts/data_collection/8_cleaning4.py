import pandas as pd
from tqdm import tqdm
from collections import defaultdict

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\df_to_analyze.csv"
final_df = pd.read_csv(path)

final_df = final_df[final_df['MIN_INT'] < 70]
final_df = final_df[final_df['PTS_home'] > 50]

RATING_THRESHOLD = 75
MAX_GAP_YEARS = 2

def fill_missing_seasons(df):

    game_cols = ['GAME_DATE_EST', 'HOME_TEAM', 'AWAY_TEAM', 'PTS_home', 'PTS_away', 'SEASON']
    schedule_df = df[game_cols].drop_duplicates(subset=['GAME_DATE_EST', 'HOME_TEAM', 'AWAY_TEAM'])

    team_schedules = defaultdict(list)

    for _, row in schedule_df.iterrows():
        season = row['SEASON']
        home = row['HOME_TEAM']
        away = row['AWAY_TEAM']

        team_schedules[(season, home)].append(row)
        team_schedules[(season, away)].append(row)

    df = df.sort_values(by=['PLAYER_NAME', 'SEASON'])
    player_groups = df.groupby('PLAYER_NAME')['SEASON'].agg(['min', 'max', 'unique'])

    new_rows_buffer = []

    for player_name, stats in tqdm(player_groups.iterrows(), total=len(player_groups), desc="Processing Players"):
        min_seas = stats['min']
        max_seas = stats['max']
        actual_seasons = set(stats['unique'])
        expected_seasons = set(range(min_seas, max_seas + 1))
        missing_seasons = sorted(list(expected_seasons - actual_seasons))

        if not missing_seasons:
            continue

        for missed_year in missing_seasons:
            prev_year = missed_year - 1

            if prev_year not in actual_seasons:
                continue

            prev_data = df[(df['PLAYER_NAME'] == player_name) & (df['SEASON'] == prev_year)].iloc[0]

            if prev_data['RATING'] < RATING_THRESHOLD:
                continue
            team_code = prev_data['PLAYER_TEAM']

            if (missed_year, team_code) in team_schedules:
                games_to_inject = team_schedules[(missed_year, team_code)]

                for game in games_to_inject:
                    new_row = {'GAME_DATE_EST': game['GAME_DATE_EST'],
                        'PTS_home': game['PTS_home'],
                        'PTS_away': game['PTS_away'],
                        'PLAYER_NAME': player_name,
                        'COMMENT': "DNP - Injury/Illness",
                        'HOME_TEAM': game['HOME_TEAM'],
                        'AWAY_TEAM': game['AWAY_TEAM'],
                        'PLAYER_TEAM': team_code,
                        'POINTS_DIFF': game['PTS_home'] - game['PTS_away'],
                        'MIN_INT': 0,
                        'SEASON': missed_year,
                        'RATING': prev_data['RATING'],
                        'AGE': prev_data['AGE'] + 1,
                        'PLAYER_HEIGHT': prev_data['PLAYER_HEIGHT'],
                        'PLAYER_WEIGHT': prev_data['PLAYER_WEIGHT'],
                        'DRAFT_YEAR': prev_data['DRAFT_YEAR'],
                        'GP': 0,
                        'PLUS_MINUS': 0,
                        'OREB_PCT': 0.0,
                        'DREB_PCT': 0.0,
                        'USG_PCT': 0.0,
                        'POS': prev_data['POS']}
                    new_rows_buffer.append(new_row)

    if new_rows_buffer:
        injected_df = pd.DataFrame(new_rows_buffer)
        final_complete_df = pd.concat([df, injected_df], ignore_index=True)

        return final_complete_df.sort_values(by=['GAME_DATE_EST', 'HOME_TEAM'])
    else:
        return df

final_df = fill_missing_seasons(final_df)
final_df = final_df.to_csv('cleaning4.csv', index=False)