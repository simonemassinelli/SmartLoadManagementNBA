import pandas as pd

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\cleaning4.csv"
df = pd.read_csv(path)

df['HOME_GAME'] = (df['PLAYER_TEAM'] == df['HOME_TEAM']).astype(int)

df.info()
print(df[['PLAYER_TEAM', 'HOME_TEAM', 'HOME_GAME']].head())

df.to_csv('cleaning4.csv', index=False)