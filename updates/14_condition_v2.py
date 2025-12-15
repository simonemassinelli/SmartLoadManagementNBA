import pandas as pd

df = pd.read_csv("../../data/initial_FE.csv")

non_injury_patterns = [
    "coach's decision",
    "personal", "suspension", "suspended", "trade",
    "rest", "d-league", "nbdl", "developmental", "family", "bereavement",
    "birth of child", "visa", "travel", "organizational", "conditioning",
    "maintenance", "ineligible", "training", "not with team",
    "simulated"
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

print(f"Rows: {len(df)}")
print(f"IS_INJURED rate: {df['IS_INJURED'].mean():.2%}")
print("\nTop injury comments:")
print(df.loc[df['IS_INJURED']==1, 'COMMENT'].value_counts().head(30))

df.to_csv('../../data/initial_FE_fixed.csv', index=False)
print("Saved!")