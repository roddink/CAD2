import xgboost as xgb
import re
from sklearn.externals import joblib
oe = joblib.load("xgb_encoder.pkl")
feature_list = oe.get_feature_names().tolist()
for ind in range(len(feature_list)):
	feature_list[ind] = re.sub("[\,\<\>\[\]]"," ",feature_list[ind])
print(len(feature_list))
dtrain = xgb.DMatrix('xgb_matrix_train.txt')#,feature_names=["placeholder"]+feature_list)
dvalid = xgb.DMatrix("xgb_matrix_valid.txt")#,feature_names=["placeholder"]+feature_list)
param = {"gamma":0,'max_depth': 10, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic',"lambda":0.01}
param['nthread'] = 16
param['eval_metric'] = 'auc'
evallist = [(dtrain, 'train'),(dvalid,"valid")]
num_round = 100
bst = xgb.train(param, dtrain, num_round, evallist)
bst.save_model('simple.model')
