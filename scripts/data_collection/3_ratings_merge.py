import pandas as pd
import glob
import os
import re

files = glob.glob("nba2k*_ratings.csv")

all_dfs = []

for f in files:
    basename = os.path.basename(f)
    match = re.search(r"nba2k(\d+)", basename.lower())
    year = match.group(1)

    df = pd.read_csv(f)
    df["Year"] = year
    all_dfs.append(df)

merged = pd.concat(all_dfs, ignore_index=True)

total_rows = len(merged)

unique_combinations = merged[['Name', 'Year']].drop_duplicates()
unique_count = len(unique_combinations)

print(f"Total rows: {total_rows}")
print(f"Unique combinations (name + season): {unique_count}")

if total_rows == unique_count:
    print("No duplicates.")
else:
    print("There are duplicates.")

final_df = merged.pivot_table(index="Name", columns="Year", values="Rating", aggfunc="first").reset_index()
final_df.columns.name = None

ordered_cols = ["Name"] + sorted([c for c in final_df.columns if c != "Name"])
final_df = final_df[ordered_cols]

rating_cols = final_df.columns.drop('Name')
final_df[rating_cols] = final_df[rating_cols].astype('Int64')

normalized_df = final_df.copy()

min_val = 50
max_val = 100

for col in rating_cols:
    valid = normalized_df[col].notna()
    pct = normalized_df.loc[valid, col].rank(pct=True) * 100
    normalized_df.loc[valid, col + "_norm"] = (min_val + (pct / 100) * (max_val - min_val)).round(1)

normalized_df.drop(rating_cols, axis=1, inplace=True)
normalized_df.to_csv("all_players_ratings_normalized.csv")

print(normalized_df.head())
print(normalized_df.loc[normalized_df['6_norm'] > 99])