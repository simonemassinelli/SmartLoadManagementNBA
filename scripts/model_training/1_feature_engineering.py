from src.features.FeatureEngineer import FeatureEngineer


dataset_path = "../../data/initial_FE_1_fixed.csv"
engineer = FeatureEngineer(dataset_path)
df_processed = engineer.engineer_all_features()

output_path = '../../data/nba_game_features_final.csv'
df_processed.to_csv(output_path, index = False)

print(f"Shape: {df_processed.shape}")
print(f"{df_processed.head()}")


