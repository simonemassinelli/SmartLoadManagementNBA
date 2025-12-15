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

non_injury_patterns = [
    "coach's decision",
    "personal",
    "suspension",
    "suspended",
    "trade",
    "rest",
    "d-league",
    "nbdl",
    "developmental",
    "family",
    "bereavement",
    "birth of child",
    "visa",
    "travel",
    "organizational",
    "conditioning",
    "maintenance",
    "ineligible",
    "training",
    "not with team",
]

injury_indicators = [
    'sore', 'sprain', 'strain', 'fracture', 'torn', 'injury',
    'knee', 'ankle', 'back', 'shoulder', 'hip', 'foot', 'hamstring',
    'calf', 'groin', 'concussion', 'illness', 'flu', 'surgery',
    'contusion', 'bruise', 'achilles', 'elbow', 'wrist', 'thigh',
    'quad', 'sick', 'ill'
]


def is_injury(comment):
    if pd.isna(comment) or str(comment).strip() == '':
        return 0

    comment_lower = str(comment).lower().strip()

    for pattern in non_injury_patterns:
        if pattern in comment_lower:
            has_injury_indicator = any(ind in comment_lower for ind in injury_indicators)
            if not has_injury_indicator:
                return 0

    return 1


df['IS_INJURED'] = df['COMMENT'].apply(is_injury)

print(f"Injury rate: {df['IS_INJURED'].mean():.2%}")

df['INJURY_HISTORY_INDEX'] = df.groupby('PLAYER_NAME')['IS_INJURED'].transform(
    lambda x: x.ewm(halflife='100 days', times=x.index).mean()
)

df = df.reset_index()

df['CONDITION'] = 100 * (1 - df['INJURY_HISTORY_INDEX'])
df['CONDITION'] = df['CONDITION'].round(1)

print(f"\nTotal rows: {len(df)}")
print(f"Unique players: {df['PLAYER_NAME'].nunique()}")

print(f"\nInjury stats:")
print(f"Total injuries: {df['IS_INJURED'].sum():,}")
print(f"Injury rate: {df['IS_INJURED'].mean():.2%}")
print(f"Players with injuries: {df[df['IS_INJURED'] == 1]['PLAYER_NAME'].nunique()}")

not_injured = df[(df['COMMENT'].notna()) & (df['IS_INJURED'] == 0)]['COMMENT'].value_counts()
print("\nClassified as NOT injury:")
print(not_injured.head(10))

injured = df[(df['COMMENT'].notna()) & (df['IS_INJURED'] == 1)]['COMMENT'].value_counts()
print("\nClassified as injury:")
print(injured.head(50))


df.to_csv("player_condition.csv", index=False)
print(f"\nSaved to player_condition.csv")