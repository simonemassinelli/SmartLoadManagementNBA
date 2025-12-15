from updates.FeatureEngineer import FeatureEngineer

input_csv = "data/initial_FE_fixed.csv" # to change path
output_csv = "data/nba_game_features_final.csv" # to change path

print(f"Feature Engineering initialization on {input_csv}")

try:
    engineer = FeatureEngineer(input_csv)

    df_final = engineer.engineer_all_features()

    df_final.to_csv(output_csv, index=False)

    print(f"\nFile saved as: {output_csv}")
    print(f"Dataset dimensions: {df_final.shape}")
    print(f"5 newest columns: {df_final.columns.tolist()[-5:]}")

except FileNotFoundError:
    print(f"ERROR, no such file {input_csv}")
except Exception as e:
    print(f"ERROR: -> {e}")