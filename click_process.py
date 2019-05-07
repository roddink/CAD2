import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

click_table = pd.read_csv("click_aug.csv")
print(click_table.describe())
nan_count = click_table.isna().sum()
for ind,col in enumerate(nan_count.keys()):
    if nan_count[col] > 10000:
        click_table.drop(columns = [col])
click_table.to_csv("click_aug_simple.csv")

