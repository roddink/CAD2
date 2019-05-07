import pandas as pd
from sklearn.externals import joblib
import json
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
data = pd.read_csv("click_aug.csv")
col_name_list =  ['objectalttext', 'objecthierarchyname', 'objectsrc', 'objectislink', 'objectisdownload', 'objecttagname', 'objecthref', 'objectid']

data = data[col_name_list]
encoder = joblib.load("click_aug_onehot_encoder3.pkl")
data = data.fillna("NAN")
# data = encoder.transform(data)
# print(data.shape)
field_list = []
print(len(field_list))
#data = data.fillna("NAN")
for ind,item in enumerate(data.values):
        field_list.append(item)
ohe =OneHotEncoder()
field_list =ohe.fit_transform(field_list)

lr_model =joblib.load("lr_model")
pred =lr_model.predict(field_list)

