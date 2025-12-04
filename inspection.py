import pandas as pd

merged = pd.read_csv(r"C:\Users\casam\OneDrive\Desktop\Simone\PycharmProjects\SmartLoadManagementNBA\first_cleaning.csv")

no_rating_df = merged[pd.isna(merged['RATING'])]

summary_no_rating = no_rating_df.groupby(['PLAYER_NAME', 'SEASON']).agg(
    games_played=('MIN_INT', 'count'),
    avg_minutes=('MIN_INT', 'mean')).reset_index()

summary_no_rating = summary_no_rating.sort_values('avg_minutes', ascending=False)

bad_pairs = set(zip(summary_no_rating['PLAYER_NAME'], summary_no_rating['SEASON']))

merged_bad = merged[merged.set_index(['PLAYER_NAME', 'SEASON']).index.isin(bad_pairs)]

total_rows = len(merged)
rows_to_remove = len(merged_bad)
remaining_rows = total_rows - rows_to_remove

print("Total rows in the dataset:", total_rows)
print("Rows to be removed:", rows_to_remove)
print("Rows to be taken:", remaining_rows)
print("\nMain players w/o rating:\n")
print(summary_no_rating.head(30))

cleaned = merged[~merged.set_index(['PLAYER_NAME', 'SEASON']).index.isin(bad_pairs)]
cleaned.to_csv('ALL_CLEANED.csv')

no_rating_new_df = cleaned[pd.isna(cleaned['RATING'])]
print(no_rating_new_df)