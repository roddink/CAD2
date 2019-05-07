import pandas as pd
import sys
from collections import defaultdict
import json
file_list = sys.argv[1:]
session_dict = defaultdict(lambda: defaultdict(list))
for file_name in file_list:
    if file_name == "click_aug.csv":
        df = pd.read_csv(file_name,nrow=7000000)
    df = pd.read_csv(file_name,nrow=7000000)
    for idx,item in enumerate(df["sessionnumber"]):
        session_dict[item][file_name].append(idx)

file_output = open("join_index_table.json","w")
json.dump(session_dict,file_output)
