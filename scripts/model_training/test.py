import pandas as pd

df = pd.read_csv("../../data/initial_FE_1_fixed.csv")
print(df["IS_INJURED"].mean())

