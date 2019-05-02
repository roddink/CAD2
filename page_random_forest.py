import pandas as pd
import sys
import page_decision_tree as pdt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pydotplus
import matplotlib.pyplot as plt

"""
    Usage : py_file_name    cleaned_page_file
"""

# file path is the path od the clean page file
file_path = sys.argv[1]
page_df_cleaned = pd.read_csv(file_path)
[x, y] = pdt.generate_train_data(page_df_cleaned)


def page_random_forest(x, y):
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
    one_rf = RandomForestClassifier(bootstrap=True, oob_score=True, max_depth=10, n_estimators=5,
                                    criterion='gini', random_state=0)
    # score = one_rf.fit(x, y)
    # headers = ["name", "score"]
    # values = sorted(zip(range(0, P), one_rf.feature_importances_), key=lambda x: x[1] * -1)
    # See which features are deemed most important by the classifier
    # print(tabulate(values, headers, tablefmt="plain"))
    # print('Random Forest OOB error rate: {}'.format(1 - one_rf.oob_score_))
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

    # bagging = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100, bootstrap=True, oob_score = True,
    #  max_samples=0.5, max_features=0.5)
    bagging = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True, oob_score=True, max_samples=0.5,
                                max_features=0.5)

    param_grid = {
        'n_estimators': n_estimators,
    }
    bagging_grid = GridSearchCV(estimator=bagging, param_grid=param_grid, cv=5, verbose=2, iid=True, n_jobs=-1)

    # Fit the grid search model
    bagging_grid.fit(x, y)

    # --------------- Ex 6 --------------- #
    # Try to change the learning rate and add it to param_grid
    # Create and fit an AdaBoosted decision tree
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate=1.0)

    learning_rate = [float(x) for x in np.linspace(0.01, 1.0, 5)]
    param_grid = {
        'n_estimators': n_estimators
    }
    boost_grid = GridSearchCV(estimator=boost, param_grid=param_grid, cv=5, verbose=2, iid=True, n_jobs=-1)

    # Fit the grid search model
    boost_grid.fit(x, y)

    # --------------- Plots --------------- #
    rf_scores = rf_grid.cv_results_['mean_test_score']
    bagging_score = bagging_grid.cv_results_['mean_test_score']
    boost_score = boost_grid.cv_results_['mean_test_score']

    plt.figure()
    plt.plot(n_estimators, rf_scores, label='Random Forest')
    plt.plot(n_estimators, bagging_score, label='Bagging')
    plt.plot(n_estimators, boost_score, label='Boosting')
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('Mean accuracy')
    plt.savefig('learning_rate_1.png')

    # Lets try and search for the optimal of two parameters for boosting
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    learning_rate = [float(x) for x in np.linspace(0.01, 1.0, 5)]
    param_grid = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }

    boost_grid = GridSearchCV(estimator=boost, param_grid=param_grid, cv=5, verbose=2, iid=True, n_jobs=-1)

    # Fit the grid search model
    boost_grid.fit(x, y)

    boost_score = boost_grid.cv_results_['mean_test_score'].reshape(len(learning_rate), len(n_estimators))

    plt.figure()
    for ind, i in enumerate(learning_rate):
        plt.plot(n_estimators, boost_score[ind], label='Learning Rate: {0:.2f}'.format(i))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('Mean accuracy')
    plt.savefig('learning_rate_2.png')


# run
page_random_forest(x, y)