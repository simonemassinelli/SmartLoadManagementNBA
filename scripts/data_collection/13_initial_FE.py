import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\df_with_standings.csv"
df = pd.read_csv(path)

df['GAME_DATE_EST']=pd.to_datetime(df['GAME_DATE_EST'])
df= df.sort_values(by=['GAME_DATE_EST', 'PLAYER_NAME'])

df['DAYS_SINCE_LG']= df.groupby(['PLAYER_NAME', 'SEASON'])['GAME_DATE_EST'].diff().dt.days.fillna(0)
df['CUMULATIVE_WL']= df.groupby(['PLAYER_NAME', 'SEASON']).cumcount()
df['START_RATIO']= df['GS_real']/ df['GP_real']
df['IS_STARTER'] = (df['START_RATIO'] > 0.5).astype(int)
df['BTB']= (df['DAYS_SINCE_LG']==1).astype(int)
def calculate_road_streak(home_game_series):
    streaks = []
    current_streak = 0
    for is_home in home_game_series:
       if is_home == 0:
           current_streak += 1
       else:
           current_streak = 0
       streaks.append(current_streak)
    return pd.Series(streaks, index=home_game_series.index)

df['ROAD_GAMES_STREAK'] = df.groupby(['PLAYER_NAME', 'SEASON'])['HOME_GAME'].transform(calculate_road_streak)

df['FIRST_SEASON_YEAR'] = df.groupby('PLAYER_NAME')['GAME_DATE_EST'].transform('min').dt.year

df.loc[df['DRAFT_YEAR'] == 'Undrafted', 'DRAFT_YEAR'] = df['FIRST_SEASON_YEAR'].astype(str)

df = df.drop(columns=['FIRST_SEASON_YEAR'])

df.info()
print(df.describe(include = 'all'))

df = df.to_csv('initial_FE.csv', index=False)