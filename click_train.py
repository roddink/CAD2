from scipy.sparse import find
import pandas as pd
from sklearn.externals import joblib
import json
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
data = pd.read_csv("click_aug.csv")
col_name_list =  [ 'objectsrc', 'objecthref']

data = data[col_name_list]
#encoder = joblib.load("click_aug_onehot_encoder3.pkl")
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
joblib.dump(ohe,"onehot_encoder.pkl")
oe = OrdinalEncoder()
#field_list = oe.fit_transform(field_list)
#joblib.dump(oe,"ordinal_encoder.pkl")
print(field_list[ind])
del data
is_cust = pd.read_csv("visitor_aug.csv")
data = is_cust["iscustomer"]
#print(len(field_list))

del is_cust
file_json = open("join_index_table2.json")
mapping = json.load(file_json)
import numpy as np
print(data.shape)
label =np.zeros((field_list.shape[0],1))
#print(len(field_list))
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
label = np.load("label_clic.npy")
lr_model.fit(field_list,label)
joblib.dump(lr_model,"lr_model2.pkl")
fpr,tpr,threshold = roc_curve(label,pred)
np.save("fpr2.npy",fpr)
np.save("tpr2.npy",tpr)
np.save('threshold2.npy',threshold)

'''
file = open("feature_click5.txt","w")
for item in new_field_list:
    file.writelines(str(item)+"\n")
file.close()
label_click = pd.DataFrame(new_y)
label_click.to_csv("label_click.csv")
'''
# joblib.dumps(file_list,"train_click_data.pkl")
