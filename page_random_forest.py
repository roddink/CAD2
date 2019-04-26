import pandas as pd
import sys
import clean_page_files as cpf
import page_decision_tree as pdt
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
import pydotplus

"""
    Usage : py_file_name    cleaned_page_file
"""

# file path is the path od the clean page file
file_path = sys.argv[1]
page_df_cleaned = cpf.read_data(file_path)
[x, y] = pdt.generate_train_data(page_df_cleaned)


def page_random_forest(x, y, data_frame_cleaned):
    [N, P] = x.shape
    clf = RandomForestClassifier(bootstrap=True, oob_score=True, criterion='gini', random_state=0, max_depth=10)
    clf.fit(x, y)
    # number of trees
    n_estimators = [int(x) for x in np.linspace(100, 500, 2)]
    # Try to add more of the parameters from the model and then add them to this dict to see how it affects the model.
    param_grid = {
        'n_estimators': n_estimators,
    }
    rf_grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=2, iid=True, n_jobs=-1)

    # Fit the grid search model
    rf_grid.fit(x, y)

    # # Look at one random forrest and the importance of the features
    one_rf = RandomForestClassifier(bootstrap=True, oob_score=True, max_depth=10, n_estimators=10,
                                    criterion='gini', random_state=0)
    score = one_rf.fit(x, y)
    headers = ["name", "score"]
    values = sorted(zip(range(0, P), one_rf.feature_importances_), key=lambda x: x[1] * -1)
    # See which features are deemed most important by the classifier
    print(tabulate(values, headers, tablefmt="plain"))
    print('Random Forest OOB error rate: {}'.format(1 - one_rf.oob_score_))
    i = 1
    for tree_in_forest in one_rf.estimators_:
        dot_data = export_graphviz(tree_in_forest,
                                   out_file=None,
                                   feature_names=x.columns,
                                   filled=True,
                                   rounded=True)
        graph_png = pydotplus.graph_from_dot_data(dot_data)
        graph_name = 'forest_tree_' + str(i) + '.pdf'
        i += 1
        graph_png.write_pdf(graph_name)
