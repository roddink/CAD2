import pandas as pd
visit_df = pd.read_csv("visitor_aug.csv")
nan_count = visit_df.isna().sum()
for col in nan_count.keys():
    if nan_count[col] > 10000:
        visit_df.drop(columns = [col])
visit_df.to_csv("visitor_aug_simple.csv")
