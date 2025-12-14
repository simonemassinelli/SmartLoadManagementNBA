import pandas as pd

path = r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\nba_game_features_final.csv"
df = pd.read_csv(path)
col_to_remove = ['BTB', 'DAYS_SINCE_LG']