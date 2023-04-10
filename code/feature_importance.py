import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def dec_tree_feat_importance(model):
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()


def log_reg_feat_importance(model):
    # Get importance
    importance = model.coef_[0]
    # Summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # Plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()


def naive_bayes_feat_importance(model, X_test, y_test):
    importance = permutation_importance(model, X_test, y_test)
    importance = importance.importances_mean
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()


def cat_boost_feat_importance(model):
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()