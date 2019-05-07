import ffm
from sklearn.externals import joblib
feature1 = open("feature_click5.txt").readlines()
ohe= joblib.load('onehot_encoder.pkl')
oe = joblib.load('ordinal_encoder.pkl')
oe_feature = []

#feature2 = open("feature_click2.txt").readlines()
for ind in range(len(feature1)):
    feature1[ind] = list(set(eval(feature1[ind].strip())))
    if ind % 10000 == 0:
        print(ind)

'''
    for item in feature1[ind]:
        temp.append([item])
    oe_feature.append(oe.transform(temp))
    if ind % 10000 == 0:
        print(ind)
'''
'''
file =open("new_feature.txt","w")
for item in oe_feature:
    file.writelines(str(item)+"\n")
file.close()
'''
'''
for ind in range(len(feature2)):
    feature2[ind] = eval(feature2[ind].strip())
feature1 = feature1+ feature2
del feature2
import pandas as pd
'''
'''
for ind in range(len(oe_feature)):
    for ind2 in range(len(oe_feature[ind])):
         oe_feature[ind][ind2] = (oe_feature[ind][ind2][0],1,1)
file =open("new_feature2.txt","w")
for item in oe_feature:
    file.writelines(str(item)+"\n")
file.close()
'''
print("read label")
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("label_click.csv")
label = data["0"].tolist()
feature_train,feature_test,label_train,label_test = train_test_split(feature1,label,test_size=0.33,random_state = 42) 
del feature1
del label
ffm_data = ffm.FFMData(feature_train,label_train)
ffm_data_valid = ffm.FFMData(feature_test,label_test)
model = ffm.FFM(eta=0.1,lam=0.0001,k=8)
model.fit(ffm_data,num_iter=500, val_data=ffm_data_valid, metric='auc', early_stopping=15, maximum=True)
model.save_model("ffm_click.bin")
