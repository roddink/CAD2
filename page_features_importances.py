import pandas as pd
import sys
import numpy as np
import page_decision_tree as pdt
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from sklearn.ensemble import ExtraTreesClassifier

"""
    Usage : py_file_name    cleaned_page_file
"""

# file path is the path od the clean page file
file_path = sys.argv[1]
page_df_cleaned = pd.read_csv(file_path)
[x, y] = pdt.generate_train_data(page_df_cleaned)


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.savefig("page_features_importance.png")
