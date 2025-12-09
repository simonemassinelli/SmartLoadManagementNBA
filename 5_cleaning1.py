import pandas as pd

merged = pd.read_csv(r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\players_games_w_ratings.csv")

no_rating_df = merged[pd.isna(merged['RATING'])]

summary_no_rating = no_rating_df.groupby(['PLAYER_NAME', 'SEASON']).agg(
    games_played=('MIN_INT', 'count'),
    avg_minutes=('MIN_INT', 'mean')).reset_index()

to_exclude = summary_no_rating[
    (summary_no_rating['games_played'] < 15) |
    (summary_no_rating['avg_minutes'] < 7)]

bad_pairs = set(zip(to_exclude['PLAYER_NAME'], to_exclude['SEASON']))
merged_filtered = merged[~merged.set_index(['PLAYER_NAME', 'SEASON']).index.isin(bad_pairs)].reset_index(drop=True)

print("Original rows:", len(merged))
print("Rows after:", len(merged_filtered))

merged_filter = merged_filtered.to_csv('first_cleaning.csv')