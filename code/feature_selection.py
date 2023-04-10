import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import GradientBoostingClassifier
from visualization import viz_correlated, viz_best_feats


def correlation(dataset, threshold, figures_path=None):
    """
    Compute feature correlation and select the correlated one
    above a threshold.
    """

    dataset = pd.DataFrame(dataset)
    col_corr = set()  
    corr_matrix = dataset.corr()

    if figures_path is not None:
        viz_correlated(corr_matrix, figures_path)

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:                 
                col_corr.add(corr_matrix.columns[i])
    return col_corr


def select_best_feats(X_train, y_train, X_test, feature_cols, cols_without_cat, figures_path=None):
    """
    Select the top ranking features given their important for the target column.
    """
    top_k =  int((2/3) * X_train.shape[1])
    bestfeatures = SelectKBest(score_func=chi2, k=top_k)
    fit_result = bestfeatures.fit(X_train[cols_without_cat], y_train)
    best_feat_df = pd.DataFrame(index = cols_without_cat)
    best_feat_df['Importance_Value'] = fit_result.scores_
    best_feat_df = best_feat_df.nlargest(top_k,'Importance_Value')
    print(best_feat_df)

    if figures_path is not None:
        viz_best_feats(X_train, best_feat_df, figures_path)

    # Select best features
    remove_features = set(cols_without_cat) - set(best_feat_df.index)
    X_train.drop(list(remove_features), axis=1, inplace=True)
    X_test.drop(list(remove_features), axis=1, inplace=True)
    feature_cols = [f for f in feature_cols if f not in remove_features]
    cols_without_cat = [f for f in cols_without_cat if f not in remove_features]

    return X_train, X_test, cols_without_cat, feature_cols
