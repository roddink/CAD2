'''
Usage: python "thisfile.py" "csv_file_name" "CategoricalVariableName1" "Categorical VariableName2"
'''
import pandas as pd
import os
import sys
from sklearn.preprocessing import OneHotEncoder#,OrdinalEncoder
from sklearn.externals import joblib
import pickle
from collections import Counter
print(sys.argv)
print(type(sys.argv[1]))
file_name = sys.argv[1]
df_cate_col = sys.argv[2:]
data = pd.read_csv(file_name)
data = data[df_cate_col]
data = data.fillna("NAN")
ohe_dict = {}
#for col in data.columns:
    #le = LabelEncoder()
    #feature = le.fit_transform(data[col])
ohe = OneHotEncoder()
ohe.fit(data)
file = open(file_name[:-4]+"_onehot_encoder3.pkl","wb")
pickle.dump(ohe,file)
