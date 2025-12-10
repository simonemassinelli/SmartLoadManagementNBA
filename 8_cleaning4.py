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


def create_dnp_row(game, player_name, team_code, ref_data, season, mode):
    age_offset = 1 if mode == "Season" else 0

    return {
        'GAME_DATE_EST': game['GAME_DATE_EST'],
        'PTS_home': game['PTS_home'],
        'PTS_away': game['PTS_away'],
        'PLAYER_NAME': player_name,
        'COMMENT': "DNP - Injury/Illness (Simulated)",
        'HOME_TEAM': game['HOME_TEAM'],
        'AWAY_TEAM': game['AWAY_TEAM'],
        'PLAYER_TEAM': team_code,
        'POINTS_DIFF': abs(game['PTS_home'] - game['PTS_away']),
        'MIN_INT': 0,
        'SEASON': season,
        'RATING': ref_data['RATING'],
        'AGE': ref_data['AGE'] + age_offset,
        'PLAYER_HEIGHT': ref_data['PLAYER_HEIGHT'],
        'PLAYER_WEIGHT': ref_data['PLAYER_WEIGHT'],
        'DRAFT_YEAR': ref_data['DRAFT_YEAR'],
        'GP': 0,
        'PLUS_MINUS': 0,
        'OREB_PCT': 0.0,
        'DREB_PCT': 0.0,
        'USG_PCT': 0.0,
        'POS': ref_data['POS']
    }


def fill_missing_games_and_seasons(df):
    game_cols = ['GAME_DATE_EST', 'HOME_TEAM', 'AWAY_TEAM', 'PTS_home', 'PTS_away', 'SEASON']
    schedule_df = df[game_cols].drop_duplicates(subset=['GAME_DATE_EST', 'HOME_TEAM', 'AWAY_TEAM'])

    team_schedules = defaultdict(list)
    for _, row in schedule_df.iterrows():
        season = row['SEASON']
        team_schedules[(season, row['HOME_TEAM'])].append(row)
        team_schedules[(season, row['AWAY_TEAM'])].append(row)

    df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE_EST'])
    unique_players = df['PLAYER_NAME'].unique()
    new_rows_buffer = []

    for player_name in tqdm(unique_players, desc="Processing"):
        player_df = df[df['PLAYER_NAME'] == player_name]

        actual_seasons = set(player_df['SEASON'].unique())
        min_seas = player_df['SEASON'].min()
        max_seas = player_df['SEASON'].max()

        expected_seasons_range = set(range(min_seas, max_seas + 1))
        missing_seasons = sorted(list(expected_seasons_range - actual_seasons))

        if missing_seasons:
            for missed_year in missing_seasons:
                prev_year = missed_year - 1
                while prev_year >= min_seas and prev_year not in actual_seasons:
                    prev_year -= 1

                if prev_year < min_seas: continue

                prev_data = player_df[player_df['SEASON'] == prev_year].iloc[-1]

                if prev_data['RATING'] < RATING_THRESHOLD:
                    continue

                team_code = prev_data['PLAYER_TEAM']

                if (missed_year, team_code) in team_schedules:
                    games = team_schedules[(missed_year, team_code)]
                    for game in games:
                        new_rows_buffer.append(
                            create_dnp_row(game, player_name, team_code, prev_data, missed_year, "Season"))

        for season in actual_seasons:
            season_data = player_df[player_df['SEASON'] == season]
            teams_in_season = season_data['PLAYER_TEAM'].unique()

            for team in teams_in_season:
                team_player_data = season_data[season_data['PLAYER_TEAM'] == team]
                ref_data = team_player_data.iloc[-1]

                if (season, team) not in team_schedules:
                    continue

                full_team_schedule = team_schedules[(season, team)]
                played_dates = set(team_player_data['GAME_DATE_EST'])

                if len(teams_in_season) > 1:
                    start_date = team_player_data['GAME_DATE_EST'].min()
                    end_date = team_player_data['GAME_DATE_EST'].max()

                    relevant_games = [
                        g for g in full_team_schedule
                        if start_date <= g['GAME_DATE_EST'] <= end_date
                    ]
                else:
                    relevant_games = full_team_schedule

                for game in relevant_games:
                    if game['GAME_DATE_EST'] not in played_dates:
                        new_rows_buffer.append(create_dnp_row(game, player_name, team, ref_data, season, "Game"))

    if new_rows_buffer:
        injected_df = pd.DataFrame(new_rows_buffer)
        final_complete_df = pd.concat([df, injected_df], ignore_index=True)
        final_complete_df = final_complete_df.drop_duplicates(subset=['PLAYER_NAME', 'GAME_DATE_EST'], keep='first')
        return final_complete_df.sort_values(by=['GAME_DATE_EST', 'HOME_TEAM'])
    else:
        return df


final_df = fill_missing_games_and_seasons(final_df)
final_df.to_csv('cleaning4.csv', index=False)