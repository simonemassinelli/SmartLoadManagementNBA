import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\cleaning4.csv"
df = pd.read_csv(path)

df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE_EST'])
df = df.set_index('GAME_DATE_EST')

has_not_played = df['COMMENT'].notna()
is_strong = df['RATING'] > 80
is_coach_dec = df['COMMENT'].str.contains("Coach's Decision", na=False, case=False)

mask_change_to_injury = has_not_played & (is_strong | (~is_strong & ~is_coach_dec))

df.loc[mask_change_to_injury, 'COMMENT'] = "DNP - Injury/Illness"

df['IS_INJURED'] = np.where(df['COMMENT'].str.contains('Injury', na=False, case=False), 1, 0)

df['INJURY_NEARBY'] = df.groupby('PLAYER_NAME')['IS_INJURED'].transform(
    lambda x: x.rolling(window=7, center=True, min_periods=1).max())

mask_weak_context_change = (~is_strong) & is_coach_dec & (df['INJURY_NEARBY'] == 1)
df.loc[mask_weak_context_change, 'COMMENT'] = "DNP - Injury/Illness"

df = df.drop(columns=['INJURY_NEARBY'])
df['IS_INJURED'] = np.where(df['COMMENT'].str.contains('Injury', na=False, case=False), 1, 0)

df['INJURY_HISTORY_INDEX'] = df.groupby('PLAYER_NAME')['IS_INJURED'].transform(
    lambda x: x.ewm(halflife='100 days', times=x.index).mean())

df = df.reset_index()

df['CONDITION'] = 100 * (1 - df['INJURY_HISTORY_INDEX'])
df['CONDITION'] = df['CONDITION'].round(1)

df_DG = df[df['PLAYER_NAME'] == 'Danilo Gallinari']

print(len(df))
print(df_DG[['GAME_DATE_EST', 'PLAYER_NAME', 'COMMENT', 'IS_INJURED', 'CONDITION']])

df.to_csv('player_condition.csv', index=False)