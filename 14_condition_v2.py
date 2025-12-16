import pandas as pd

df = pd.read_csv(r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\initial_FE.csv")
df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
df = df.sort_values(['PLAYER_NAME', 'GAME_DATE_EST'])

non_injury_patterns = [
    "coach's decision", "personal", "suspension", "suspended", "trade",
    "rest", "d-league", "nbdl", "developmental", "family", "bereavement",
    "birth of child", "visa", "travel", "organizational", "conditioning",
    "maintenance", "ineligible", "training", "not with team",
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
    s = str(comment).lower().strip()
    if any(p in s for p in non_injury_patterns):
        return 0
    return int(any(ind in s for ind in injury_indicators))

df['IS_INJURED'] = df['COMMENT'].apply(is_injury)

df['PREV_INJURED'] = df.groupby('PLAYER_NAME')['IS_INJURED'].shift(1).fillna(0)
df['NEW_INJURY'] = ((df['IS_INJURED'] == 1) & (df['PREV_INJURED'] == 0)).astype(int)

df['INJURED_NEXT_GAME'] = df.groupby('PLAYER_NAME')['NEW_INJURY'].shift(-1).fillna(0)

print(f"IS_INJURED rate: {df['IS_INJURED'].mean():.2%}")
print(f"NEW_INJURY rate: {df['NEW_INJURY'].mean():.2%}")
print(f"INJURED_NEXT_GAME rate: {df['INJURED_NEXT_GAME'].mean():.2%}")

df.to_csv('initial_FE_fixed.csv', index=False)
print("Saved!")