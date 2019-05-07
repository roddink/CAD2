import pickle 
file_feature = open('combine_feature.pkl','rb')
label_feature = open('combine_label.pkl','rb')
feature_dict = pickle.load(file_feature)
label_dict = pickle.load(label_feature)
page_id = []
object_id = []
object_name = []
for ind,key in enumerate(feature_dict):
	temp ={key:[]}    
	for item in feature_dict[key]:
        	temp[key].append(item[1],item[2])
'''	
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
page_id_encoder = LabelEncoder()
object_id_encoder = LabelEncoder()
object_name_encoder = LabelEncoder()
page_id_encoder.fit(page_id)
object_id_encoder.fit(object_id)
object_name_encoder.fit(object_name)
joblib.dump(page_id_encoder,'page_id_encoder.pkl')
joblib.dump(object_id,'object_id_encoder.pkl')
joblib.dump(object_name,'object_name_encoder.pkl')
'''
