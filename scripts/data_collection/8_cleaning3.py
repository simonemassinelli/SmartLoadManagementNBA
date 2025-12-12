import pandas as pd
import numpy as np

path1 = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\simone_cleaned.csv"
path2 = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\DS_with_Ratings.csv"

prev = pd.read_csv(path1)
final_df = pd.read_csv(path2)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

final_df = final_df.sort_values(['PLAYER_NAME', 'SEASON'])

cols_to_fill = ['PLAYER_HEIGHT', 'PLAYER_WEIGHT', 'DRAFT_YEAR', 'GP', 'NET_RATING', 'OREB_PCT', 'DREB_PCT', 'USG_PCT', 'POS']

final_df[cols_to_fill] = final_df.groupby('PLAYER_NAME')[cols_to_fill].ffill()
final_df['AGE'] = final_df['AGE'].fillna(final_df.groupby('PLAYER_NAME')['AGE'].shift(1) + 1)

final_df[cols_to_fill] = final_df.groupby('PLAYER_NAME')[cols_to_fill].bfill()
final_df['AGE'] = final_df['AGE'].fillna(final_df.groupby('PLAYER_NAME')['AGE'].shift(-1) - 1)

avg_pg = final_df.loc[final_df['POS'] == 'PG', 'PLAYER_HEIGHT'].mean()
avg_sg = final_df.loc[final_df['POS'] == 'SG', 'PLAYER_HEIGHT'].mean()
avg_sf = final_df.loc[final_df['POS'] == 'SF', 'PLAYER_HEIGHT'].mean()
avg_pf = final_df.loc[final_df['POS'] == 'PF', 'PLAYER_HEIGHT'].mean()
avg_c = final_df.loc[final_df['POS'] == 'C', 'PLAYER_HEIGHT'].mean()

t1 = (avg_pg + avg_sg) / 2
t2 = (avg_sg + avg_sf) / 2
t3 = (avg_sf + avg_pf) / 2
t4 = (avg_pf + avg_c) / 2


def get_pos(row):
    if pd.notna(row['POS']):
        return row['POS']

    if pd.isna(row['PLAYER_HEIGHT']):
        return 'SF'

    h = row['PLAYER_HEIGHT']

    if h <= t1:
        return 'PG'
    elif h <= t2:
        return 'SG'
    elif h <= t3:
        return 'SF'
    elif h <= t4:
        return 'PF'
    else:
        return 'C'


final_df['POS'] = final_df.apply(get_pos, axis=1)


def clean_plus_minus(row):
    val = row['PLUS_MINUS']

    if pd.isna(val):
        return 0

    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


final_df['PLUS_MINUS'] = final_df.apply(clean_plus_minus, axis=1)

cols_to_int = ['PTS_home', 'PTS_away', 'POINTS_DIFF', 'RATING', 'AGE', 'GP', 'PLUS_MINUS']

for col in cols_to_int:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

final_df[cols_to_int] = final_df[cols_to_int].fillna(0)
final_df[cols_to_int] = final_df[cols_to_int].astype(int)

final_df = final_df.drop('GS', axis=1)

final_df = final_df[final_df['MIN_INT']<66]

final_df['NET_RATING_real'] = final_df.groupby(['PLAYER_NAME', 'SEASON'])['PLUS_MINUS'].transform('mean')
final_df = final_df.drop('NET_RATING', axis = 1)

final_df.info()
print(final_df.describe(include='all'))

final_df.to_csv('df_to_analyze.csv', index=False)