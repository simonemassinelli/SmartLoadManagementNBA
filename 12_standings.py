import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

path1 = r"C:\Users\casam\.cache\kagglehub\datasets\bluedreamv1b3\nba-teams-stat-2000-2023\versions\1\advanced_stats_total.csv"
path2 = r"C:\Users\casam\.cache\kagglehub\datasets\bluedreamv1b3\nba-teams-stat-2000-2023\versions\1\division_total_E.csv"
path3 = r"C:\Users\casam\.cache\kagglehub\datasets\bluedreamv1b3\nba-teams-stat-2000-2023\versions\1\division_total_W.csv"
path4 = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\player_condition.csv"

df_standings = pd.read_csv(path1)
df_wpct_est = pd.read_csv(path2)
df_wpct_west = pd.read_csv(path3)
df_player = pd.read_csv(path4)

df_wpct_est = df_wpct_est.rename(columns={'Eastern_Conference': 'Team'})
df_wpct_west = df_wpct_west.rename(columns={'Western_Conference': 'Team'})

nba_team_map = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
    'New Jersey Nets': 'BKN',
    'Seattle SuperSonics': 'OKC',
    'Charlotte Bobcats': 'CHA',
    'New Orleans Hornets': 'NOP',
    'New Orleans/Oklahoma City Hornets': 'NOP',
    'Vancouver Grizzlies': 'MEM'}

def clean_team_names(df):
    if 'Rk' in df.columns:
        df = df[df['Rk'] != 'Rk']
    df = df[df['Team'] != 'League Average']
    df['Team'] = df['Team'].str.replace('*', '', regex=False).str.strip()
    return df

df_standings = clean_team_names(df_standings)
df_wpct_est = clean_team_names(df_wpct_est)
df_wpct_west = clean_team_names(df_wpct_west)

df_divisions = pd.concat([df_wpct_est, df_wpct_west], ignore_index=True)
df_divisions = df_divisions[['Year', 'Team', 'W/L_percent']]

df_merged_standings = pd.merge(df_standings, df_divisions, on=['Year', 'Team'], how='left')
df_merged_standings['TEAM_ABBR'] = df_merged_standings['Team'].map(nba_team_map)

df_merged_standings['YEAR_2DIGIT'] = df_merged_standings['Year'] - 2000

cols_to_keep = ['TEAM_ABBR', 'YEAR_2DIGIT', 'Rk', 'W/L_percent']
df_final_standings = df_merged_standings[cols_to_keep]
df_final_standings = df_final_standings.rename(columns={'W/L_percent': 'WIN_PCT'})

df_player = pd.merge(
    df_player,
    df_final_standings,
    left_on=['SEASON', 'HOME_TEAM'],
    right_on=['YEAR_2DIGIT', 'TEAM_ABBR'],
    how='left')

df_player = df_player.rename(columns={'Rk': 'HOME_TEAM_POSITION', 'WIN_PCT': 'HOME_TEAM_W_PCT'})
df_player = df_player.drop(columns=['YEAR_2DIGIT', 'TEAM_ABBR'])

df_player = pd.merge(
    df_player,
    df_final_standings,
    left_on=['SEASON', 'AWAY_TEAM'],
    right_on=['YEAR_2DIGIT', 'TEAM_ABBR'],
    how='left')

df_player = df_player.rename(columns={'Rk': 'AWAY_TEAM_POSITION', 'WIN_PCT': 'AWAY_TEAM_W_PCT'})
df_player = df_player.drop(columns=['YEAR_2DIGIT', 'TEAM_ABBR'])

print(df_player[['GAME_DATE_EST', 'HOME_TEAM', 'HOME_TEAM_POSITION', 'HOME_TEAM_W_PCT',
                 'AWAY_TEAM', 'AWAY_TEAM_POSITION', 'AWAY_TEAM_W_PCT']].head(20))

df_player.to_csv('df_with_standings.csv', index=False)