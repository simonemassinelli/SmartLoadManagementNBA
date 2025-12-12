import kagglehub

path1 = kagglehub.dataset_download("nathanlauga/nba-games")
print("Path to dataset files:", path1)

path2 = kagglehub.dataset_download("bluedreamv1b3/nba-teams-stat-2000-2023")
print("Path to dataset files:", path2)