import pandas as pd
import sys
import clean_page_files as cpf
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import warnings
warnings.filterwarnings('ignore')

"""
    Usage : py_file_name    cleaned_page_file
"""

# file path is the path od the clean page file
file_path = sys.argv[1]
page_df_cleaned = cpf.read_data(file_path)


def generate_train_data(data_frame_cleaned):
    """
    Generate the data for the decision tree
    :param data_frame_cleaned: the data from representing the file page already cleaned
    :return: features x and the response variable iscustomer
    """
    data_frame_cleaned = data_frame_cleaned.drop(columns="referringpageinstanceid")
    y = data_frame_cleaned["iscustomer"]
    cont_names = list(["sessionnumber", "pageinstanceid", "eventtimestamp", "pagesequenceinsession"])
    x_page_location_domain_dummies = pd.get_dummies(page_df_cleaned["pagelocationdomain"])
    x = pd.concat([data_frame_cleaned[cont_names], x_page_location_domain_dummies], axis=1)
    return x, y


def page_decision_tree(x, y, data_frame_cleaned):
    d_tree = DecisionTreeClassifier()
    # Casting response into categorical, as decision trees expects cat response
    d_tree.fit(x, y)
    levels_list = list(np.unique(data_frame_cleaned["pagelocationdomain"]))
    features_names = levels_list.extend(levels_list)
    # we export the tree as a graphviz format
    dot_data = export_graphviz(dtree, out_file=None,
                               feature_names=features_names,
                               class_names='iscustomer',
                               filled=True, rounded=True,
                               special_characters=True)
    graph_png = pydotplus.graph_from_dot_data(dot_data)
    graph_png.write_pdf('page_tree.pdf')


[x, y] = generate_train_data(page_df_cleaned)
page_decision_tree(x, y, page_df_cleaned)