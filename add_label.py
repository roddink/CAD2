import pandas as pd 
data_visitor = pd.read_csv("visitor_aug.csv",error_bad_lines = False,encoding="latin-1")
iscust = data_visitor["iscustomer"]
data_click = pd.read_csv("click_aug.csv",error_bad_lines = False,encoding ='latin-1')
import pickle
import json
import numpy as np
data_is_cust = np.zeros((data_click.shape[0],1))
data_is_cust[:] = np.nan
file = open("join_index_table2.json")
mapping = json.load(file)
data_is_cust = np.zeros((len(mapping),1))
data_is_cust[:] = np.nan
from collections import defaultdict
feature_dict = defaultdict(list)
labels = defaultdict(int)
for item in mapping:
    item2 = mapping[item]
    labels[item] = iscust[item2['visitor_aug.csv'][0]]
    for num_page in item2.get("click_aug.csv",[]):
        feature_dict[item].append((data_click['pageinstanceid'][num_page],data_click['objectid'],data_click['objectname']))
file_new = open('combine_feature.pkl','wb')
pickle.dump(feature_dict,file_new)
file_new2 = open('combine_.pkl','wb')
pickle.dump(labels,file_new2)
