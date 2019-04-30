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


def lines_to_delete(data_frame, cont_names):
    """
    Clean the data for the values which have the wrong type
    :param data_frame: pandas data frame pre-cleaned
    :param cont_names: names of the continuous variables
    :return: dictionary which contains the index of the rows to delete
    """

    df_size = data_frame.shape[0]
    to_drop_dic = dict.fromkeys(cont_names, None)
    for col in cont_names:
        for i in range(df_size):
            if not isinstance(data_frame[col].iloc[i], np.int64):
                to_drop_dic[col] = i
    return to_drop_dic


def delete_noise(data_frame, to_drop_dic):
    """
    drop the wrong values in the continuous variable before transforming them as dummies
    :param data_frame: pandas data frame pre-cleaned
    :param to_drop_dic: dictionary which contains the index of the rows to delete
    :return: data frame cleaned
    """
    col_names = list(to_drop_dic.keys())
    for col in col_names:
        data_frame = data_frame.drop(data_frame.index[np.asarray(to_drop_dic[col])])
    return data_frame


def generate_train_data(data_frame_cleaned):
    """
    Generate the data for the decision tree
    :param data_frame_cleaned: the data from representing the file page already cleaned
    :return: features x and the response variable iscustomer
    """
    y = data_frame_cleaned["iscustomer"]
    cont_names = list(["sessionnumber", "pageinstanceid", "eventtimestamp", "pagesequenceinsession"])
    for col in cont_names:
        data_frame_cleaned = data_frame_cleaned.drop(data_frame_cleaned.loc[:, [isinstance(data_frame_cleaned[col]
                                                                                           , str)]])
    x_page_location_domain_dummies = pd.get_dummies(page_df_cleaned["pagelocationdomain"])
    x = pd.concat([data_frame_cleaned[cont_names], x_page_location_domain_dummies], axis=1)
    return x, y


def page_decision_tree(x, y, data_frame_cleaned):
    """
    Generate the decision tree of the page file
    :param x: features
    :param y: response variable
    :param data_frame_cleaned: data frame of the initial data cleaned
    :return: a decision tree
    """
    d_tree = DecisionTreeClassifier()
    # Casting response into categorical, as decision trees expects cat response
    d_tree.fit(x, y)
    features_names = list(np.unique(data_frame_cleaned["pagelocationdomain"]))
    # we export the tree as a graphviz format
    dot_data = export_graphviz(dtree, out_file=None,
                               feature_names=features_names,
                               class_names='iscustomer',
                               filled=True, rounded=True,
                               special_characters=True)
    graph_png = pydotplus.graph_from_dot_data(dot_data)
    graph_png.write_pdf('page_tree.pdf')


# read the data for the tree
[x, y] = generate_train_data(page_df_cleaned)
page_decision_tree(x, y, page_df_cleaned)