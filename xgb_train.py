import pandas as pd
from sklearn.externals import joblib
import json
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

data = pd.read_csv("click_aug.csv")
col_name_list = ['objectalttext', 'objecthierarchyname', 'objectsrc', 'objectislink', 'objectisdownload',
                 'objecttagname', 'objecthref', 'objectid']
col_name_list = ["objectalttext", "objectsrc", "objecthref"]
data_key = data["sessionnumber"]
data = data[col_name_list]
data = data.fillna("NAN")
field_list = []
print(len(field_list))
for ind, item in enumerate(data.values):
    field_list.append(item)
ohe = OneHotEncoder()
field_list = ohe.fit_transform(field_list)
joblib.dump(ohe,"xgb_encoder.pkl")
#del data
#is_cust = pd.read_csv("visitor_aug.csv")
#data_ = is_cust["iscustomer"]
import numpy as np
'''
label = np.load("label_clic.npy")
new_file = open("xgboost_process.txt","w")
for ind,item in enumerate(field_list):
    if ind % 10000 ==0:
        print(ind)
    if data.values[ind][0] != "NAN" or data.values[ind][1] != "NAN" or data.values[ind][2] != "NAN":
        new_file.writelines(str(int(data_key.values[ind])) + " ")
        for item_index,item_data in zip(item.indices,item.data):
            new_file.writelines(str(int(item_index))+":"+str(int(item_data))+ " ")
        new_file.writelines("\n")
new_file.close()

'''

